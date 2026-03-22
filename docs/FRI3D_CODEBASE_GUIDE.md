# FRi3D Codebase Guide

This document is a human-readable introduction to the FRi3D implementation in this repository. It explains:

- the high-level architecture,
- how the paper equations map to code,
- the step-by-step data flow from model parameters to synthetic measurements,
- and where fitting/optimization lives.

The reference paper is:
**Isavnin (2016), "FRiED: A Novel Three-dimensional Model of Coronal Mass Ejections"**.

---

## 1. High-Level Overview

The repository implements a **3D analytical CME flux-rope model** with two main usage modes:

- **Static mode**: one snapshot of a CME geometry + magnetic field.
- **Dynamic mode**: time-dependent CME parameters for synthetic in-situ time series.

At a high level, the model pipeline is:

1. Define global CME parameters (size, orientation, flattening, twist, flux, etc.).
2. Build an axis curve \(r(\phi)\) and cross-section scaling.
3. Generate a 3D shell (optionally with pancaking and skew).
4. Populate shell with twisted magnetic field lines.
5. Sample synthetic spacecraft measurements at arbitrary points/trajectories.
6. Optionally fit model parameters to in-situ data using differential evolution.

Core file:

- `src/ai/fri3d/model.py` (geometry, magnetic field, static/dynamic models)

Supporting files:

- `src/ai/fri3d/lib.pyx` (Cython integrands used by `scipy.integrate.quad`)
- `src/ai/fri3d/optimize.py` (fitting functions and profile classes)
- `examples/` (usage examples)

---

## 2. Repository Structure (What Each File Does)

- `src/ai/fri3d/model.py`
  - `BaseFRi3D`: shared parameter interface.
  - `StaticFRi3D`: one-time snapshot model.
  - `DynamicFRi3D`: time-dependent wrapper around static snapshot generation.
  - Geometry, field line construction, synthetic in-situ sampling.

- `src/ai/fri3d/lib.pyx`
  - Low-level mathematical integrands (axis length, flux-related integrals).
  - Used to speed up repeated numerical integration.

- `src/ai/fri3d/optimize.py`
  - `fit2insitu`: fit dynamic FRi3D parameters to in-situ magnetic field and speed.
  - `fit2cor`: visual overlay for coronagraph images.
  - `BaseProfile`, `PolyProfile`, `ExpProfile`, `SignProfile` for time profiles.

- `src/ai/fri3d/differentialevolution.py`
  - Vendored/custom differential evolution solver.
  - Currently `fit2insitu` uses SciPy's solver directly.

- `examples/`
  - Scripts for shell, line, map, data, and fitting usage.

---

## 3. Equation-to-Code Mapping (Paper -> Implementation)

The most important mappings are:

- **Eq. (14)** axis shape \(r(\phi)=R_t \cos^n(a\phi)\)
  - `StaticFRi3D.vanilla_axis_height`

- **Eq. (1)** cross-section radius scaling \(R(\phi)=\frac{R_p}{R_t}r(\phi)\)
  - Implemented when tapering shell/lines using axis height and poloidal height.

- **Eq. (15)** axis length integral
  - `vanilla_axis_length` + Cython `vanilla_axis_dlength`.

- **Eq. (17)-(19)** magnetic flux conservation and pitch-angle-related terms
  - Numerical integral in `line` using Cython `modded_axis_1Dfluxintgrand`.

- **Eq. (20), Eq. (21)** dynamic growth laws for \(R_t(t)\), \(R_p(t)\)
  - Implemented generically in `DynamicFRi3D` via user-supplied time profiles.

- **Eq. (5), Eq. (11)** normal-angle/curvature-related geometry
  - Used implicitly in normal-angle and differential geometry terms.

---

## 4. Step-by-Step Implementation Walkthrough

### Step 1: Parameter Definition and Validation

`BaseFRi3D` defines model properties:

- orientation: `latitude`, `longitude`, `tilt`
- geometry: `toroidal_height`, `half_width`, `half_height`, `flattening`, `pancaking`, `skew`
- magnetic: `twist`, `flux`, `sigma`, `polarity`, `chirality`

`StaticFRi3D` setters validate ranges and derive internal values:

- `_coeff_angle = pi/(2*half_width)`
- `_poloidal_height = toroidal_height*tan(half_height)`

Angles are wrapped with `subtract_period`.

---

### Step 2: Axis Geometry (Backbone of CME)

The axis in polar coordinates is implemented by:

\[
r(\phi)=R_t\cos^n\left(\frac{\pi}{2\phi_{hw}}\phi\right)
\]

This is the paper's Eq. (14), implemented in `vanilla_axis_height`.

Other axis helpers:

- `vanilla_axis_distance`: distance from a spatial point to a chosen axis point.
- `vanilla_axis_min_distance`: finds closest axis point via scalar optimization.
- `vanilla_axis_normal_angle`: normal/radial angle derived from axis slope.
- `vanilla_axis_length`: arc length via numerical integration.

---

### Step 3: Shell Construction (`shell`)

`shell(phi, theta)` builds 3D surface points of the CME.

Execution order:

1. Sample axis coordinate `phi` and cross-section angle `theta`.
2. Compute axis height + local normal angle.
3. Start from tapered cylindrical cross-sections (Eq. (1) scaling idea).
4. Apply **pancaking**: deform circular cross-section to ellipse-like shape.
5. Bend local cross-sections onto the axis curve.
6. Apply global orientation rotation (`latitude`, `longitude`, `tilt`).
7. Apply **skew** azimuthal deformation:
   \[
   \phi \leftarrow \phi + \phi_s\left(1-\frac{r}{R_t}\right)
   \]

Output: `(x, y, z)` arrays of shell points.

---

### Step 4: Magnetic Field Line Construction (`line`)

`line(r, phi, theta)` returns one twisted field line and field magnitude.

Pipeline:

1. Compute twist phase accumulation along axis using integrated axis-height term.
2. Apply chirality sign (`+1` right-handed, `-1` left-handed).
3. Apply tapering + pancaking to cross-section radii.
4. Compute local axial field amplitude by enforcing flux consistency numerically.
5. Use radial field profile controlled by `sigma`.
6. Bend/orient/skew exactly as in shell pipeline.

Returns: `(x, y, z, b)` where `b` is scalar field magnitude along the line.

Note:

- The paper discusses Lundquist-style field strength in Eq. (16), then numerical handling after deformations.
- The implementation uses a compact Gaussian-like radial profile parameterized by `sigma` plus numerical flux normalization.

---

### Step 5: Synthetic In-Situ Measurement (`data`)

`data(x, y, z)` predicts what a spacecraft measures at one or many points.

This function inverts geometry first, then reconstructs local field:

1. Reverse skew.
2. Reverse global orientation.
3. Convert to the model's local frame/cylindrical representation.
4. Find nearest axis point for each query point.
5. Compute local normalized radius \(r_{rel}\) in deformed cross-section.
6. Reverse local twist phase.
7. If \(r_{rel}\le 1\): sample line at \(\phi\pm d\phi\) and use tangent as field direction.
8. Scale by local magnitude and polarity.

Outputs:

- magnetic field vectors `B`
- two speed coefficients used by dynamic model (`vtc`, `vpc`)

Points outside the flux-rope boundary return NaNs.

---

### Step 6: Dynamic Evolution (`DynamicFRi3D`)

`DynamicFRi3D` stores each parameter as a callable `p(t)`.

Main methods:

- `snapshot(t)`: evaluate all profiles at time `t` and create/update static snapshot.
- `insitu(t, x, y, z)`: evaluate synthetic in-situ series over time.
- `impact(t, x, y, z)`: find minimum approach distance over a time interval.

Relation to paper Eq. (20)-(21):

- The code is more general than fixed linear laws.
- If user supplies linear profiles, it reproduces Eq. (20)/(21)-style evolution.

---

### Step 7: Numerical Integration and Cython Kernels (`lib.pyx`)

Cython functions used by `quad`:

- `vanilla_axis_height`
- `vanilla_axis_dlength`
- `modded_axis_height`
- `modded_axis_1Dfluxintgrand`

These speed up repeated integration during:

- axis length estimation,
- twist accumulation,
- flux-consistent magnetic amplitude computation.

---

### Step 8: Fitting and Optimization (`optimize.py`)

### `fit2insitu`

Goal: fit dynamic FRi3D profiles to measured in-situ data.

How it works:

1. Build `DynamicFRi3D`.
2. Attach profile objects for each parameter (polynomial/exponential/sign).
3. Define residual:
   - generate synthetic `B(t)` and `V(t)`,
   - compare to observations using FastDTW-based distance in a normalized feature space.
4. Optimize free profile parameters using differential evolution.

Returns:

- fitted `DynamicFRi3D`
- fitted profile dictionary

### `fit2cor`

- Produces visual overlays of model shell on coronagraph images.
- Useful for inspection/initialization, but not a full automated image inverse problem.

---

## 5. Coordinate Systems and Dependencies

The code heavily uses `ai.cs` for:

- Cartesian/spherical/cylindrical conversions,
- rotation matrices and matrix application.

This means coordinate correctness depends strongly on `ai.cs` behavior.

---

## 6. Practical End-to-End Usage Paths

### Static visualization

1. Create `StaticFRi3D(...)`.
2. Call `shell(...)` for wireframe.
3. Call `line(...)` to draw field lines.

### Synthetic spacecraft pass

1. Create `StaticFRi3D(...)`.
2. Build trajectory `(x(t), y(t), z(t))`.
3. Call `data(...)` to get `B`.

### Time-evolving event

1. Create `DynamicFRi3D(...)` with callable profiles.
2. Call `insitu(...)` over timestamps.
3. Optionally call `fit2insitu(...)` to infer profile parameters from data.

---

## 7. Notes and Caveats

- Some example scripts still use legacy argument names (for example `poloidal_height`) and may need adaptation to current `StaticFRi3D` signature.
- The local vendored differential evolution implementation exists, but fitting currently uses SciPy's DE.
- No substantial test suite is currently present in `tests/`.
- `forcemap` uses `numdifftools`, which is not listed in `requirements.txt`.

---

## 8. Quick Orientation for New Contributors

If you are new to this codebase, read in this order:

1. `src/ai/fri3d/model.py` (`StaticFRi3D` first, then `DynamicFRi3D`)
2. `src/ai/fri3d/lib.pyx` (integrand definitions)
3. `examples/example_shell.py`, `examples/example_data.py`
4. `src/ai/fri3d/optimize.py` (`fit2insitu` and profile classes)

That path gives the fastest route from model physics to runnable output.
