"""
Example: Using SpinQICK driver with Stanza

This example demonstrates how to integrate SpinQICK's DCSource voltage control
system with Stanza's instrument framework.

Prerequisites:
1. Install: pip install cq-stanza[spinqick]
2. Have a SpinQICK hardware_config.json file configured
3. Have a voltage source that implements SpinQICK's VoltageSource protocol

For more details, see SPINQICK_INTEGRATION.md
"""

from stanza.drivers.spinqick import SpinQick
from stanza.models import BaseInstrumentConfig, InstrumentType
from stanza.base.channels import ChannelConfig
from stanza.models import PadType, GateType

# Example: Create a mock voltage source for testing
# In production, this would be your actual QDAC, Basel LNHR, etc.
class MockVoltageSource:
    """Mock voltage source for demonstration"""

    def open(self, address: str):
        print(f"Opening connection to {address}")

    def close(self):
        print("Closing connection")

    def get_voltage(self, ch: int) -> float:
        print(f"Getting voltage from channel {ch}")
        return 0.0

    def set_voltage(self, ch: int, volts: float):
        print(f"Setting channel {ch} to {volts}V")

    def set_sweep(self, ch: int, start: float, stop: float, length: float, num_steps: int):
        print(f"Programming sweep on channel {ch}: {start}V -> {stop}V in {num_steps} steps")

    def trigger(self, ch: int):
        print(f"Triggering channel {ch}")

    def arm_sweep(self, ch: int):
        print(f"Arming sweep on channel {ch}")


def main():
    """Main example demonstrating SpinQICK driver usage"""

    # Step 1: Create voltage source
    # In production, use your actual voltage source:
    # from my_hardware import QDevil_QDAC
    # voltage_source = QDevil_QDAC()
    voltage_source = MockVoltageSource()

    # Step 2: Configure the instrument
    instrument_config = BaseInstrumentConfig(
        name="spinqick_control",
        type=InstrumentType.GENERAL,
        serial_addr="192.168.1.100",
        port=5025,
    )

    # Step 3: Configure channels
    # IMPORTANT: Channel names must match gate names in hardware_config.json
    channel_configs = {
        "P1": ChannelConfig(
            name="P1",  # Must match SpinQICK gate name
            voltage_range=(-0.5, 0.5),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        ),
        "P2": ChannelConfig(
            name="P2",
            voltage_range=(-0.5, 0.5),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=2,
        ),
        "M1": ChannelConfig(
            name="M1",  # Sensor gate
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.SENSOR,
            control_channel=3,
        ),
    }

    # Step 4: Create SpinQICK instrument
    print("Initializing SpinQICK instrument...")
    spinqick = SpinQick(
        instrument_config=instrument_config,
        channel_configs=channel_configs,
        voltage_source=voltage_source,
    )

    print(f"\nCreated: {spinqick}")
    print(f"Control channels: {spinqick.control_channels}")

    # Step 5: Basic voltage control
    print("\n=== Basic Voltage Control ===")
    spinqick.set_voltage("P1", 0.5)
    voltage = spinqick.get_voltage("P1")
    print(f"P1 voltage: {voltage}V")

    # Step 6: Cross-coupling compensation
    print("\n=== Cross-Coupling Compensation ===")
    # Set P1 while keeping M1 (sensor) constant
    spinqick.set_voltage_compensated("P1", 0.6, iso_gates=["M1"])
    print("Set P1 to 0.6V with M1 compensation")

    # Step 7: Voltage ramps
    print("\n=== Voltage Ramps ===")
    spinqick.program_ramp(
        channel_name="P2",
        vstart=0.0,
        vstop=0.1,
        tstep=10e-6,  # 10 microseconds per step
        nsteps=100
    )
    spinqick.arm_sweep("P2")
    spinqick.trigger_sweep("P2")
    print("Executed voltage ramp on P2")

    # Step 8: State management
    print("\n=== State Management ===")
    all_voltages = spinqick.get_all_voltages()
    print(f"All voltages: {all_voltages}")

    # Save voltage state
    spinqick.save_voltage_state("./voltage_state_example.yaml")
    print("Saved voltage state to file")

    # Step 9: Channel-level control
    print("\n=== Channel-Level Control ===")
    p1_channel = spinqick.get_channel("control_P1")
    print(f"P1 gate name: {p1_channel.gate_name}")
    p1_channel.set_voltage_compensated(0.7, iso_gates=["M1"])
    print("Set P1 to 0.7V via channel interface")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    # Note: This example requires:
    # 1. SpinQICK to be installed
    # 2. A valid hardware_config.json file
    # In production, you would also need actual hardware connected

    try:
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install SpinQICK:")
        print("  pip install cq-stanza[spinqick]")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("  1. SpinQICK is installed")
        print("  2. hardware_config.json is properly configured")
        print("  3. Gate names in channel_configs match hardware_config.json")
