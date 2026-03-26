# Betaflight -- Flight Controller Reference

This document summarizes the betaflight project as it relates to elodin SITL simulation.

## Overview

Betaflight is an open-source flight controller firmware for multirotor aircraft. The version used here is a **custom fork** (branch `skypulse`) based on Betaflight 4.5.0-RC2, with 67 custom commits adding features like dual VTX, multi-RX, and license system.

**Location**: `/Users/eugenedudnik/Projects/Tebenko/Betaflight/betaflight`
**Remote**: `git@git.noblebyte.com:Airplane/FC/betaflight.git`
**Branch**: `skypulse` (default)
**Also available as submodule**: `examples/betaflight-sitl/betaflight`

## Project Structure

```
betaflight/
  Makefile              # Build system (default target: STM32F405)
  src/main/
    main.c              # Entry point
    fc/                 # Flight controller core: PID loop (core.c), arming, failsafe, RC
    flight/             # Flight dynamics: mixer, PID, GPS rescue, altitude hold, servos
    rx/                 # Receiver protocols: CRSF, SBUS, MSP, Spektrum, etc.
    sensors/            # Sensor drivers: gyro, accel, baro, compass, GPS
    drivers/            # HAL: SPI, I2C, UART, timers, VTX, pinio
    io/                 # I/O: serial, beeper, LEDs, VTX, OSD, fuse.c/fuse.h (custom)
    config/             # EEPROM config storage, license.c/license.h (custom)
    msp/                # MultiWii Serial Protocol
    osd/                # On-Screen Display
    cli/                # Serial command-line interface
    target/             # Board/target definitions (14 targets)
      SITL/             # Software-In-The-Loop target
  obj/main/             # Build output (betaflight_SITL.elf)
  help/                 # Documentation (changes.md, elodin.md)
```

## SITL Target

The SITL (Software-In-The-Loop) target allows running betaflight as a regular process on the host machine, communicating with the simulator via UDP/TCP.

### Building

```bash
cd /Users/eugenedudnik/Projects/Tebenko/Betaflight/betaflight
make TARGET=SITL
# Output: obj/main/betaflight_SITL.elf

# With debug symbols and optional features:
make TARGET=SITL DEBUG=GDB OPTIONS="USE_ALTITUDE_HOLD USE_GPS"
```

### SITL Architecture

**Key files** in `src/main/target/SITL/`:
- `sitl.c` (750 lines) -- Core: UDP/TCP networking, system init, virtual time, PWM output, virtual EEPROM
- `target.h` (293 lines) -- Defines, packet structures (`fdm_packet`, `rc_packet`, `servo_packet`)
- `udplink.c` -- UDP socket abstraction

**Characteristics**:
- Uses pthreads for concurrent UDP/TCP handling (3 worker threads)
- Virtual time with `simRate` scaling -- advances based on simulation delta vs wall-clock delta
- Virtual sensors: gyro, accel, baro, compass (via virtual drivers)
- `USE_IMU_CALC` is undefined -- attitude comes directly from simulator quaternion, not AHRS
- PID loop rate: 10 kHz virtual (100 us period)
- EEPROM backed by `eeprom.bin` file (32 KB)
- 8 virtual UARTs (TCP-backed via dyad library)
- Default RX provider: MSP

### Communication Ports

| Port | Protocol | Direction | Data |
|------|----------|-----------|------|
| UDP 9001 | Raw PWM | BF -> Sim | Raw PWM values |
| UDP 9002 | Normalized | BF -> Sim | Motor commands (0.0-1.0) |
| UDP 9003 | FDM | Sim -> BF | Sensor data (144 bytes) |
| UDP 9004 | RC | Sim -> BF | RC channel values (40 bytes) |
| TCP 5761 | MSP | Bidirectional | Configuration, joystick |

### Motor Remapping

Betaflight SITL applies a Gazebo-compatible remap in `pwmCompleteMotorUpdate()` (sitl.c). After remapping, the indices match the Quad-X configuration used in elodin:

```
motor[0] = Front Right (FR, CCW, spin +1)
motor[1] = Back Left  (BL, CCW, spin +1)
motor[2] = Front Left  (FL, CW, spin -1)
motor[3] = Back Right  (BR, CW, spin -1)
```

### Packet Structures

```c
// FDM packet (simulator -> betaflight, 144 bytes)
struct fdm_packet {
    double timestamp;           // simulation time
    double imu_angular_velocity_rpy[3];  // gyro (rad/s, FRD)
    double imu_linear_acceleration_xyz[3]; // accel (m/s², FRD)
    double imu_orientation_quat[4];      // quaternion (w,x,y,z)
    double velocity_xyz[3];              // world velocity (m/s)
    double position_xyz[3];              // world position (m)
};

// Servo/motor packet (betaflight -> simulator, 16 bytes)
struct servo_packet {
    float motor_speed[4];  // normalized 0.0 - 1.0
};
```

## Key Custom Features (skypulse fork)

Features relevant to simulation:
- **Advanced Failsafe** -- 3 failsafe stage 2 modes including attitude hold. May affect SITL behavior if RC signal is lost.
- **FUSE & CHARGE Modes** -- PinIO operational modes with 45s post-arm safety timer. Not active in SITL but affects arming logic.
- **Multi-RX Support** -- Up to 3 simultaneous receivers. In SITL, only MSP RX is used.

Features not relevant to simulation (hardware-only):
- Dual VTX Support, License System, OSD Enhancements, Extended PINIO, Compass Calibration

## Key Source Locations for Debugging

| File | What to look for |
|------|-----------------|
| `src/main/fc/core.c` | PID loop, main control flow |
| `src/main/flight/mixer.c` | Motor mixing, output scaling |
| `src/main/target/SITL/sitl.c` | SITL networking, time management, motor output |
| `src/main/fc/rc.c` | RC channel processing |
| `src/main/fc/rc_controls.c` | Arming logic, mode activation |
| `src/main/sensors/gyro.c` | Gyro processing pipeline |
| `src/main/io/serial.c` | Serial port management |

## Integration with Elodin

1. Elodin's s10 process orchestrator spawns the SITL binary
2. `BetaflightSyncBridge` (comms.py) implements lockstep:
   - Send FDM sensor packet (UDP 9003) + RC channels (via MSP on TCP 5761)
   - Block until motor command response (UDP 9002)
3. Coordinate conversion: Elodin uses ENU/FLU, Betaflight uses FRD -- conversion in sensors.py
4. Betaflight SITL binary path: `../betaflight/obj/main/betaflight_SITL.elf` (top-level betaflight fork, not a submodule)
5. EEPROM config: `../betaflight/eeprom.bin` (created on first SITL run, configure via CLI or Configurator)

## Upstream Status

- Fork base: Betaflight 4.5.0-RC2
- Upstream now at: 2025.12.2
- Notable upstream additions since fork: native altitude hold, position hold, collision detection
- Merge conflict risk: HIGH for RX handlers and VTX, MEDIUM for OSD and CLI, LOW for config defaults
- Additional branches: `skypulse-4.6-final`, `simulator-4.6`, `althold-2` for porting work
