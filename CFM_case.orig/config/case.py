"""
Case configuration for the 2D Euler-Euler bubble-plume solver.

Everything physical and numerical lives here as a dataclass. Edit this file
for a permanent change to the case; run.py can also override these values
for a single run without touching this file.

x is the channel gap, x=0 at the electrode. y is the channel height, y=0 at
the bottom. Gravity acts along -y.

Phase 1 is the liquid (KOH electrolyte), phase 2 the dispersed gas (H2).
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Geometry:
    L: float = 0.005        # channel gap (width), m
    H: float = 0.5           # channel height, m


@dataclass
class Mesh:
    nx: int = 60              # cells across the gap
    ny: int = 240             # cells along the height

    # x_grading_mode: "electrode" refines near x=0 only, "both" mirrors the
    # refinement at x=L too (needed once the far wall is close enough to matter).
    x_grading_mode: str = "electrode"
    # Near-electrode cell widths [m], e.g. (20e-6, 30e-6, 50e-6). The rest of
    # the gap grows geometrically from the last one. Empty -> uniform mesh.
    x_first_cells: tuple = ()


@dataclass
class Fluids:
    # Liquid: 6 M KOH electrolyte
    rho_l: float = 1190.0           # kg/m^3
    mu_l: float = 1.95e-3           # kg/(m s)
    # Gas: hydrogen, density from ideal gas at p=1 bar, T=298.15 K
    mu_g: float = 8.9e-6            # kg/(m s)
    M_H2: float = 2.016e-3          # kg/mol, molar mass of H2
    d_b: float = 100.0e-6           # bubble diameter, m

    # 6 M KOH at 298 K: approximately 0.073 N/m (close to pure water)
    sigma: float = 0.073            # N/m, liquid-gas surface tension

    p_op: float = 1.0e5             # Pa
    T_op: float = 298.15            # K
    R: float = 8.314                # J/(mol K)

    @property
    def rho_g(self) -> float:
        return self.p_op * self.M_H2 / (self.R * self.T_op)

    @property
    def V_m(self) -> float:
        return self.R * self.T_op / self.p_op


@dataclass
class GasSource:
    """Electrolytic gas generation, applied in the near-electrode cells.
    See src/source.py and the report for the mass-source formula."""
    j: float = 277.7               # current density, A/m^2  (= 0.5 A/cm^2)
    F: float = 96485.33             # Faraday constant, C/mol
    n_electrode_cells: int = 1      # near-wall cell columns that emit gas
    electrode_y_start: float = 0.0  # active electrode extent, fraction of H
    electrode_y_end: float = 0.9
    ramp_time: float = 0.1          # s, linear ramp-up of the source


@dataclass
class Turbulence:
    kappa: float = 0.41             # von Karman constant
    A_plus: float = 26.0            # van Driest damping constant
    sigma_d: float = 0.7            # turbulent Schmidt number (dispersion)
    C_mu_bit: float = 0.6           # Sato bubble-induced turbulence coefficient
    C_hydro_disp: float = 1.0       # hydrodynamic dispersion coefficient, see alpha_transport.py
    bit_sato: bool = False          # Sato bubble-induced turbulence on/off
    nut_max_factor: float = 1.0e4   # cap nu_t at this multiple of nu_l
    model: str = "laminar"          # "laminar" (nu_t=0) | "mixing_length"


@dataclass
class Numerics:
    t_end: float = 1.5              # s, physical end time
    dt_init: float = 2.0e-4         # s, initial time step
    cfl: float = 0.25               # target Courant number
    diffusive_number: float = 0.60  # fraction of the local explicit FV bound
    dt_max: float = 1.0e-3          # s
    dt_min: float = 1.0e-9          # s, hard floor for the rejection controller

    # Adaptive dt takes the largest stable explicit step, then retries at a
    # smaller dt if the step blows up (non-finite, or too fast a velocity).
    dt_adaptive_reject: bool = True
    dt_reject_shrink: float = 0.5   # shrink dt by this on a rejected step
    dt_grow: float = 1.1            # grow the headroom by this after a clean step
    dt_max_retries: int = 6
    dt_reject_umax: float = 1.0     # m/s; reject if any phase speed exceeds this

    n_corr: int = 2                 # PISO pressure-drag corrector iterations
    alpha_max: float = 0.95         # numerical cap on gas volume fraction
    relax_p: float = 0.7            # pressure under-relaxation
    p_ref_value: float = 0.0        # reference (gauge) pressure at the outlet
    save_every: float = 0.05        # s, interval between field dumps
    print_every: int = 50           # iterations between console reports

    # n_outer=1 is pure PISO; n_outer>1 re-linearises convection and drag each
    # outer iteration (PIMPLE) and relies on relax_U to stay stable.
    n_outer: int = 1
    relax_U: float = 1.0

    drag_model: str = "Tomiyama_contaminated"  # informational, see interfacial.py
    drag_alpha_floor: float = 1.0e-4  # numerical floor used only in drag coupling
    dispersion: bool = False        # turbulent dispersion of alpha, D_t=nu_t/sigma_d
    dispersion_hydro: bool = False  # bubble-induced (hydrodynamic) dispersion
    convection: str = "upwind"      # "upwind" or "limited" (van Leer TVD)

    gas_visc_cap_factor: float = 1.0  # caps gas momentum diffusivity at this * nu_l; <=0 uncapped

    # Reuses the pressure matrix's LU factorisation across correctors when the
    # coefficients haven't changed -- the usual case, and the cheap speed-up.
    reuse_pressure_lu: bool = True


@dataclass
class Inlet:
    """Bottom inlet. U_in<=0 -> closed wall; U_in>0 -> liquid enters upward.
    alpha_g_in is normally 0 (no gas injected from the bottom)."""
    U_in: float = 0.0              # m/s, upward bottom inlet velocity
    alpha_g_in: float = 0.0        # gas fraction imposed at bottom inlet

    @property
    def active(self) -> bool:
        return self.U_in > 0.0


@dataclass
class SolidPressure:
    """Johnson-Jackson-type collisional pressure, applied everywhere as a
    force on the gas phase. See interfacial.py and the report for p_coll."""
    gamma0: float = 0.002
    gamma1: float = 2.0
    gamma2: float = 5.0
    alpha_min: float = 0.0
    alpha_jj_max: float = 0.65
    alpha_clip_eps: float = 1.0e-4
    accel_cap: float = 1250.0       # m/s^2 cap on gas accel near alpha_jj_max; <=0 disables
    reaction_on_liquid: bool = False


@dataclass
class Post:
    """Recording controls for time-resolved post-processing."""
    probe_dt: float = 2.0e-3        # s, sampling interval for time series
    snap_dt: float = 0.05           # s, interval between alpha snapshots
    probe_heights: tuple = (0.25, 0.5, 0.75, 0.9)  # fractions of H
    probe_x_frac: float = 0.2       # fraction of L for the spectrum probe


@dataclass
class Case:
    geom: Geometry = field(default_factory=Geometry)
    mesh: Mesh = field(default_factory=Mesh)
    fluids: Fluids = field(default_factory=Fluids)
    source: GasSource = field(default_factory=GasSource)
    turb: Turbulence = field(default_factory=Turbulence)
    num: Numerics = field(default_factory=Numerics)
    solid_pressure: SolidPressure = field(default_factory=SolidPressure)
    post: Post = field(default_factory=Post)
    inlet: Inlet = field(default_factory=Inlet)

    g: float = 9.81                 # gravitational acceleration, m/s^2

    @property
    def gy(self) -> float:
        return -self.g


CASE = Case()
