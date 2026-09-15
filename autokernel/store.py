from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from pathlib import Path

from .integrity import atomic_write, canonical, read_json, write_json


class Store:
    """Supervisor-owned local storage. Workers never receive its signing key."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.objects = self.root / "objects"
        self.objects.mkdir(exist_ok=True)
        key_path = self.root / "supervisor.key"
        try:
            fd = os.open(key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as stream:
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
        if key_path.stat().st_mode & 0o077:
            raise PermissionError("supervisor key must be owner-only")
        self.key = key_path.read_bytes()
        if len(self.key) != 32:
            raise ValueError("invalid supervisor signing key")
        self.db = sqlite3.connect(self.root / "state.sqlite", timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY, campaign TEXT NOT NULL, spec TEXT NOT NULL,
                state TEXT NOT NULL, attempt TEXT, receipt TEXT, started REAL);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, at REAL NOT NULL, payload TEXT NOT NULL,
                previous TEXT NOT NULL, signature TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS budgets (
                campaign TEXT PRIMARY KEY, attempts INTEGER NOT NULL DEFAULT 0,
                spent REAL NOT NULL DEFAULT 0, reserved REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS reservations (
                attempt TEXT PRIMARY KEY, campaign TEXT NOT NULL, amount REAL NOT NULL,
                settled INTEGER NOT NULL DEFAULT 0);
        """)

    def close(self):
        self.db.close()

    @contextlib.contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        else:
            self.db.execute("COMMIT")

    def put(self, content: bytes) -> str:
        identity = hashlib.sha256(content).hexdigest()
        path = self.objects / identity
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError("corrupted content-addressed object")
        else:
            atomic_write(path, content)
        return identity

    def get(self, identity) -> bytes:
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("invalid object identity")
        content = (self.objects / identity).read_bytes()
        if hashlib.sha256(content).hexdigest() != identity:
            raise ValueError("artifact digest mismatch")
        return content

    def sign(self, payload):
        return hmac.new(self.key, canonical(payload), hashlib.sha256).hexdigest()

    def envelope(self, payload):
        return {"payload": payload, "signature": self.sign(payload)}

    def verify(self, envelope):
        if set(envelope) != {"payload", "signature"}:
            raise ValueError("invalid signed envelope")
        if not hmac.compare_digest(self.sign(envelope["payload"]), envelope["signature"]):
            raise ValueError("receipt signature mismatch")
        return envelope["payload"]

    def event(self, payload):
        previous = self.db.execute("SELECT signature FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        at = time.time()
        record = {"at": at, "payload": payload, "previous": previous[0] if previous else ""}
        self.db.execute("INSERT INTO events(at,payload,previous,signature) VALUES (?,?,?,?)",
                        (at, canonical(payload).decode(), record["previous"], self.sign(record)))

    def verify_events(self):
        import json

        previous = ""
        for row in self.db.execute("SELECT * FROM events ORDER BY seq"):
            record = {"at": row["at"], "payload": json.loads(row["payload"]), "previous": previous}
            if row["previous"] != previous or not hmac.compare_digest(row["signature"], self.sign(record)):
                raise ValueError("event chain corrupted")
            previous = row["signature"]

    def freeze(self, campaign):
        path = self.root / "campaign.json"
        value = campaign.model_dump(mode="json")
        if path.exists() and self.verify(read_json(path)) != value:
            raise ValueError("campaign is immutable; use a new store")
        if not path.exists():
            write_json(path, self.envelope(value))
        return campaign.identity

    def enqueue(self, campaign, candidate):
        import json

        with self.transaction():
            row = self.db.execute("SELECT campaign,spec FROM candidates WHERE id=?", (candidate.identity,)).fetchone()
            if row:
                if row["campaign"] != campaign or json.loads(row["spec"]) != candidate.model_dump(mode="json"):
                    raise ValueError("candidate identity conflict")
                return False
            self.db.execute("INSERT INTO candidates(id,campaign,spec,state) VALUES (?,?,?,?)",
                            (candidate.identity, campaign, canonical(candidate.model_dump(mode="json")).decode(),
                             "proposed"))
            self.event({"type": "enqueued", "candidate": candidate.identity})
        return True

    def claim(self, campaign, candidate_id, budget, reservation):
        if not 0 < reservation <= budget.max_wall_seconds:
            raise ValueError("invalid budget reservation")
        attempt = secrets.token_hex(16)
        with self.transaction():
            self.db.execute("INSERT OR IGNORE INTO budgets(campaign) VALUES (?)", (campaign,))
            state = self.db.execute("SELECT * FROM budgets WHERE campaign=?", (campaign,)).fetchone()
            if state["attempts"] >= budget.max_candidates:
                raise RuntimeError("candidate budget exhausted")
            if state["spent"] + state["reserved"] + reservation + budget.reserve_confirmation_seconds > budget.max_wall_seconds:
                raise RuntimeError("wall-time budget exhausted")
            row = self.db.execute("SELECT state,campaign FROM candidates WHERE id=?", (candidate_id,)).fetchone()
            if not row or row["state"] != "proposed" or row["campaign"] != campaign:
                raise RuntimeError("candidate is not available")
            self.db.execute("UPDATE candidates SET state='running',attempt=?,started=? WHERE id=?",
                            (attempt, time.time(), candidate_id))
            self.db.execute("UPDATE budgets SET attempts=attempts+1,reserved=reserved+? WHERE campaign=?",
                            (reservation, campaign))
            self.db.execute("INSERT INTO reservations(attempt,campaign,amount) VALUES (?,?,?)",
                            (attempt, campaign, reservation))
            self.event({"type": "claimed", "candidate": candidate_id, "attempt": attempt})
        return attempt

    def finish(self, candidate_id, attempt, receipt, elapsed):
        if not 0 <= elapsed < float("inf"):
            raise ValueError("invalid elapsed time")
        payload = receipt.model_dump(mode="json")
        if payload["candidate_digest"] != candidate_id or payload["attempt_id"] != attempt:
            raise ValueError("receipt identity mismatch")
        identity = self.put(canonical(self.envelope(payload)))
        passed = payload["gates"].get("numerical", {}).get("status") == "passed"
        state = "numerical_pass" if passed and not payload["error"] else "rejected"
        with self.transaction():
            row = self.db.execute("SELECT state,attempt,campaign FROM candidates WHERE id=?", (candidate_id,)).fetchone()
            if not row or row["state"] != "running" or row["attempt"] != attempt:
                raise RuntimeError("stale or duplicate attempt completion")
            if payload["campaign_digest"] != row["campaign"]:
                raise ValueError("wrong campaign receipt")
            reservation = self.db.execute("SELECT * FROM reservations WHERE attempt=?", (attempt,)).fetchone()
            self.db.execute("UPDATE candidates SET state=?,receipt=? WHERE id=?", (state, identity, candidate_id))
            self.db.execute("UPDATE budgets SET reserved=reserved-?,spent=spent+? WHERE campaign=?",
                            (reservation["amount"], elapsed, row["campaign"]))
            self.db.execute("UPDATE reservations SET settled=1 WHERE attempt=?", (attempt,))
            self.event({"type": "completed", "candidate": candidate_id, "attempt": attempt,
                        "state": state, "receipt": identity})
        return identity

    def recover(self):
        """Call only while holding the controller lock after old workers have stopped."""
        with self.transaction():
            rows = self.db.execute("SELECT * FROM candidates WHERE state='running'").fetchall()
            for row in rows:
                reservation = self.db.execute("SELECT * FROM reservations WHERE attempt=?", (row["attempt"],)).fetchone()
                # Conservatively charge the entire reservation; a crash must not replenish budgets.
                self.db.execute("UPDATE budgets SET reserved=reserved-?,spent=spent+? WHERE campaign=?",
                                (reservation["amount"], reservation["amount"], row["campaign"]))
                self.db.execute("UPDATE reservations SET settled=1 WHERE attempt=?", (row["attempt"],))
                self.db.execute("UPDATE candidates SET state='interrupted' WHERE id=?", (row["id"],))
                self.event({"type": "interrupted", "candidate": row["id"], "attempt": row["attempt"]})
            orphaned = self.db.execute("SELECT * FROM reservations WHERE settled=0").fetchall()
            for reservation in orphaned:
                self.db.execute("UPDATE budgets SET reserved=reserved-?,spent=spent+? WHERE campaign=?",
                                (reservation["amount"], reservation["amount"], reservation["campaign"]))
                self.db.execute("UPDATE reservations SET settled=1 WHERE attempt=?", (reservation["attempt"],))
                self.event({"type": "interrupted_activity", "attempt": reservation["attempt"]})
        return len(rows)

    @contextlib.contextmanager
    def activity(self, campaign, budget, *, seconds, kind):
        """Reserve a bounded diagnostic/training operation; charge elapsed time even on failure."""
        if not 0 < seconds <= budget.max_wall_seconds:
            raise ValueError("invalid activity budget")
        attempt = secrets.token_hex(16)
        with self.transaction():
            self.db.execute("INSERT OR IGNORE INTO budgets(campaign) VALUES (?)", (campaign,))
            row = self.db.execute("SELECT * FROM budgets WHERE campaign=?", (campaign,)).fetchone()
            if row["spent"] + row["reserved"] + seconds > budget.max_wall_seconds:
                raise RuntimeError("activity exceeds remaining wall-time budget")
            self.db.execute("UPDATE budgets SET reserved=reserved+? WHERE campaign=?", (seconds, campaign))
            self.db.execute("INSERT INTO reservations(attempt,campaign,amount) VALUES (?,?,?)", (attempt, campaign, seconds))
            self.event({"type": "activity_reserved", "kind": kind, "attempt": attempt, "seconds": seconds})
        started = time.monotonic()
        try:
            yield attempt
        finally:
            with self.transaction():
                self.db.execute("UPDATE budgets SET reserved=reserved-?,spent=spent+? WHERE campaign=?",
                                (seconds, time.monotonic() - started, campaign))
                self.db.execute("UPDATE reservations SET settled=1 WHERE attempt=?", (attempt,))
                self.event({"type": "activity_settled", "attempt": attempt})

    def rows(self):
        return [dict(row) for row in self.db.execute("SELECT * FROM candidates ORDER BY rowid")]

    def receipt(self, identity):
        import json

        return self.verify(json.loads(self.get(identity)))

    def storage_bytes(self):
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
