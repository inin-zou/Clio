"""Telephony glue: Twilio inbound webhook → LiveKit room mint → Modal spawn.

This package is the only place backend code runs business logic at call setup
time (before the control-plane WS exists). After the webhook returns, control
moves to backend/app/control/orchestrator.py.
"""
