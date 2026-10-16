"""Dependency-free wire limits defined by the current ExoCode firmware source."""

# ExoCode.ino MotorParams::T_MAX and clampTorque both define ±18 Nm.
# This is the protocol ceiling, not an operator default or an enable intent.
TORQUE_MAX_NM = 18.0
