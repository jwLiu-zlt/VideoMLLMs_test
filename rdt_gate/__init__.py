"""Fast-Slow RDT-Gate proof-of-concept package."""

from .decision_schema import GateDecision
from .prototype_bank_gate import PrototypeBankGate
from .prototype_gate import Signal

__all__ = ["GateDecision", "PrototypeBankGate", "Signal"]
