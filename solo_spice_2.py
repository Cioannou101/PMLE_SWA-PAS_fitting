
# %%
import numpy as np
import spiceypy as spice
from scipy import constants as sc
from datetime import datetime


def get_radial_distance(start_time, day_span):

    # Change current working directory to Project2
    # project_dir = '/disk/plasma2/cio/Project2'
    # os.chdir(project_dir)

    ########################################################################
    ####### Load up all the spice kernels that you need.
    ########################################################################

    spice.furnsh('solo_spice/solo_ANC_soc-sci-fk_V08.tf')              # Frames kernel
    spice.furnsh('solo_spice/pck00010.tpc')                           # Planetary constants
    spice.furnsh('solo_spice/naif0012.tls')                           # Leap second kernel
    spice.furnsh('solo_spice/solo_ANC_soc-sclk_20250803_V01.tsc')    # SC clock kernel
    spice.furnsh('solo_spice/de421.bsp')                              # Generic planet ephemerides

    # The .bsp kernel is the actual data that tells you where SolO is.
    # Keep an eye on this as they get updated by SOC occasionally.

    spkFile='solo_spice/solo_ANC_soc-orbit_20200210-20301120_L022_V1_00464_V01.bsp'
    spice.furnsh(spkFile) # SO ephermerides

    Vr = sc.au * 1e-3
    # span = day_span * 24 * 60 * 60
    # et0 = spice.str2et(epoch_start)

    span = day_span * 24 * 60 * 60
    et0 = spice.str2et(start_time)

    ets = et0 + np.arange(span)  # all times in seconds past J2000

    sc_pos = np.zeros((4, span))
    times_utc = []

    for ii, et in enumerate(ets):
        dState, _ = spice.spkezr('SOLO', et, 'SOLO_HCI', 'NONE', 'Sun')
        sc_pos[0:3, ii] = dState[0:3]
        sc_pos[3, ii] = np.linalg.norm(dState[0:3])
        times_utc.append(spice.et2utc(et, 'C', 2))

    SOx, SOy, SOz = sc_pos[0] / Vr, sc_pos[1] / Vr, sc_pos[2] / Vr
    radialDist = np.sqrt(SOx**2 + SOy**2 + SOz**2)

    spice.unload('solo_spice/solo_ANC_soc-sci-fk_V08.tf')              # Frames kernel
    spice.unload('solo_spice/pck00010.tpc')                           # Planetary constants
    spice.unload('solo_spice/naif0012.tls')                           # Leap second kernel
    spice.unload('solo_spice/solo_ANC_soc-sclk_20250803_V01.tsc')    # SC clock kernel
    spice.unload('solo_spice/de421.bsp')                              # Generic planet ephemerides
    spice.unload(spkFile) # SO ephermerides

    return radialDist, np.array(times_utc)

def precompute_times1_grid(times1_str):
    """
    Precompute the start timestamp and length for a perfect per-second grid.
    """
    fmt = "%Y %b %d %H:%M:%S.%f"
    start_dt = datetime.strptime(times1_str[0], fmt)
    start_ts = start_dt.timestamp()
    length = len(times1_str)
    return start_ts, length

def find_indices_perfect_grid(times1_str, times2_dt):
    """
    Compute closest indices for a perfect per-second grid.
    """
    start_ts, length = precompute_times1_grid(times1_str)
    times2_ts = np.array([t.timestamp() for t in times2_dt], dtype=np.float64)
    idxs = np.rint(times2_ts - start_ts).astype(int)  # round to nearest second
    return np.clip(idxs, 0, length - 1)  # ensure within range

def to_year_doy_string(date_tuple):
    year, month, day = map(int, date_tuple[:3])  # First three as strings → ints
    hour, minute, second = date_tuple[3:]        # Last three already ints
    dt = datetime(year, month, day, hour, minute, second)
    doy = dt.timetuple().tm_yday
    return f"{year}-{doy:03d}T{hour:02d}:{minute:02d}:{second:02d}"

def get_radial_distance_PAS(date_str, day_span, times_pas):

    start_time = to_year_doy_string(date_str)

    R, time = get_radial_distance(start_time, day_span)

    indices = find_indices_perfect_grid(time, times_pas)

    R_pas = R[indices]

    return R_pas
