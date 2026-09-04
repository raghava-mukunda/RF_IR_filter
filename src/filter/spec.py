from dataclasses import dataclass


@dataclass(frozen=True)
class FilterSpec:
    """
    Cryogenic microwave low-pass filter design specification.

    All frequencies are in Hz.
    Impedance is in ohms.
    """

    z0: float = 50.0

    passband_edge: float = 9e9
    passband_ripple_db: float = 0.1

    stopband_start: float = 20e9
    stopband_attenuation_db: float = 60.0

    analysis_start: float = 10e6
    analysis_stop: float = 70e9

    def validate(self) -> None:
        if self.z0 <= 0:
            raise ValueError("Z0 must be positive.")

        if self.passband_edge <= 0:
            raise ValueError("Passband edge must be positive.")

        if self.stopband_start <= self.passband_edge:
            raise ValueError(
                "Stopband start must be above passband edge."
            )

        if self.passband_ripple_db <= 0:
            raise ValueError(
                "Passband ripple must be positive."
            )

        if self.stopband_attenuation_db <= 0:
            raise ValueError(
                "Stopband attenuation must be positive."
            )

        if self.analysis_start <= 0:
            raise ValueError(
                "Analysis start must be positive."
            )

        if self.analysis_stop <= self.stopband_start:
            raise ValueError(
                "Analysis stop must extend beyond stopband."
            )


SPEC = FilterSpec()


if __name__ == "__main__":
    SPEC.validate()

    print("Cryogenic filter specification")
    print("--------------------------------")
    print(f"Z0                 = {SPEC.z0:.1f} ohm")
    print(f"Passband edge      = {SPEC.passband_edge / 1e9:.3f} GHz")
    print(f"Passband ripple    = {SPEC.passband_ripple_db:.3f} dB")
    print(f"Stopband start     = {SPEC.stopband_start / 1e9:.3f} GHz")
    print(f"Stopband attenuation >= {SPEC.stopband_attenuation_db:.1f} dB")
    print(f"Analysis range     = {SPEC.analysis_start / 1e9:.3f}"
          f" - {SPEC.analysis_stop / 1e9:.1f} GHz")
