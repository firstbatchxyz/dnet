"""Multiprocessing context fakes for profile_subproc tests."""

from __future__ import annotations

import json


class FakeMPConn:
    def __init__(self, payload):
        self._payload = payload
        self._closed = False

    def send(self, data):
        self._payload = data

    def recv(self):
        return self._payload

    def close(self):
        self._closed = True


class FakeMPProc:
    def __init__(self, target, args, daemon):
        self._target = target
        self._args = args
        self.daemon = daemon
        self.exitcode = None

    def start(self):
        conn = self._args[
            -1
        ]  # simulate a child that writes an error and exits successfully
        conn.send(json.dumps({"_error": "fail"}))
        self.exitcode = 0

    def join(self, timeout):
        return

    def terminate(self):
        self.exitcode = 1


class FakeMPContext:
    def Pipe(self, duplex=False):
        payload = json.dumps({"_error": "fail"})
        parent = FakeMPConn(payload)
        child = FakeMPConn(payload)
        return parent, child

    def Process(self, target, args, daemon):
        return FakeMPProc(target, args, daemon)
