# Numerical method

## Equations

Reduced Guyer–Krumhansl model, non-dimensional:

    ∂t T + (Kn²/3) ∇·h = 0
    Φ ∂t h + h + Φ ∇T − Kn² Φ ∇·(Φ ∇h) = 0,     Φ = max(0, 1 − Kn²|h|²/6)

In the full model (`full_q=True`) the last term is replaced by `Φ ∇·Q`, with
`ε Φ ∂t Q + Q + Kn² Φ ∇h = 0` and `ε = τ_Q/τ_R`.

The system has the structure of a Stokes–Brinkman problem in which `T`
plays the role of the pressure and `h` of the velocity.  It is discretised
on a staggered (MAC) grid.

## Boundary conditions

| boundary | conditions |
|---|---|
| `x = 0` | `T = T_hot`, `h_y = 0` |
| `x = L` | `T = 0`, `h_y = 0` |
| `y = 0, R` | `h_y = 0`, and `∂h_x/∂y = 0` (free slip) or `h_x = 0` (no slip, channel tests) |
| obstacle | `h = 0` |

Nothing else is imposed.  No condition on `T` is needed on the walls or on
the obstacle, because `T` is differentiated only at interior heat-flux
nodes.  At `x = 0, L` the viscous term needs `∂h_x/∂x`: evaluating the
energy equation on the boundary, where `T` is fixed and `h_y ≡ 0`, gives
`∂h_x/∂x = ∇·h = −(3/Kn²) ∂t T = 0`, which is used as the normal flux.

## Grid

`dx = L/Nx`, `dy = R/Ny`, arrays indexed `[j, i]`:

| field | position | shape |
|---|---|---|
| `T` | `(i dx, (j+½) dy)` | `(Ny, Nx+1)` |
| `h_x` | `((i+½) dx, (j+½) dy)` | `(Ny, Nx)` |
| `h_y` | `(i dx, j dy)` | `(Ny+1, Nx+1)` |
| corners | `((i+½) dx, j dy)` | `(Ny+1, Nx)` |

Each `T` node is the centre of the energy control volume
`[x_{i−½}, x_{i+½}] × [y_j, y_{j+1}]`, whose faces carry
`h_x[j, i−1], h_x[j, i], h_y[j, i], h_y[j+1, i]`.  The Dirichlet values of `T`
lie on the columns `i = 0, Nx`, and the condition `h_y = 0` on the boundary
rows and columns of `h_y`: every boundary datum sits on a node.

## Interior discretisation

Energy, at the interior `T` nodes:

    dT/dt = −(Kn²/3) [ (h_x[j,i] − h_x[j,i−1])/dx + (h_y[j+1,i] − h_y[j,i])/dy ]

Heat flux, at the non-solid `h_x` and interior `h_y` nodes:

    dh_x/dt = −h_x/Φ_x − (T[j,i+1] − T[j,i])/dx + Kn² D_x
    dh_y/dt = −h_y/Φ_y − (T[j,i]   − T[j−1,i])/dy + Kn² D_y

The viscous term `D = ∇·(Φ∇h)` is written in flux form:

    F_xx = Φ_T ∂x h_x,  F_yy = Φ_T ∂y h_y    on the T nodes
    F_yx = Φ_C ∂y h_x,  F_xy = Φ_C ∂x h_y    on the corners
    D_x = δx F_xx + δy F_yx,    D_y = δx F_xy + δy F_yy

with two-point differences throughout; for `Φ = 1` it reduces to the
five-point Laplacian of each component.  `Φ` is evaluated from the local
`|h|²` wherever it is needed (`Φ_x`, `Φ_y`, `Φ_T`, `Φ_C`), the missing
component being obtained by two- or four-point averaging; in the relaxation
term it is bounded below by `phi_floor = 1e-3`.

In the full model `Q_xx, Q_yy` are stored on the `T` nodes and `Q_yx, Q_xy`
on the corners, i.e. where the fluxes `F` live.  Each component relaxes to
`−Kn² F` with the exact factor `exp(−dt/(ε Φ))`, and `div Q` uses the same
differences as `D`, so that the full scheme reduces exactly to the reduced
one as `ε → 0`.

Properties:

* gradient and divergence are compact two-point differences, and the
  discrete gradient is minus the adjoint of the discrete divergence.  The
  scheme therefore has no odd–even (checkerboard) mode, and in the linear
  case the discrete energy `(3/2Kn²) Σ T² + ½ Σ |h|²` cannot grow through the
  T–h coupling;
* the energy balance is conservative: summed over all control volumes the
  interior fluxes cancel.

## Wall conditions in discrete form

* `x = 0, L`: `T[:, 0] = T_hot`, `T[:, Nx] = 0`, not evolved.  The first
  `h_x` node sees the centred gradient `(T[:, 1] − T_hot)/dx`.
  `h_y[:, 0] = h_y[:, Nx] = 0`.  The flux `F_xx` on the boundary `T` nodes is
  zero; since the `h_x` control volume next to the boundary is `[0, dx]`,
  this is a finite-volume Neumann condition.
* `y = 0, R`: `h_y[0, :] = h_y[Ny, :] = 0`, so the first row of energy
  volumes receives no flux through the wall.  The `h_x` rows are half a cell
  from the wall; the ghost value `h_x^g = h_x` (free slip) or `−h_x` (no slip)
  gives `F_yx` on the wall corners.

## Obstacle

The obstacle is the closed square `S`.  The whole treatment is a mask on the
heat-flux nodes:

1. every `h_x` and `h_y` node in `S` is set to zero and never updated.  The
   normal component imposes `h·n = 0` through the fluxes of the adjacent
   energy volumes, the tangential component imposes `h·t = 0` through the
   viscous stencil of the neighbouring fluid nodes;
2. the energy equation is applied unchanged at every `T` node.  A `T` node
   whose four faces are solid keeps its initial value and is never read by
   any fluid equation, since a fluid `h` node only reads the two `T` nodes of
   which it is a face.  No extension of `T` into the body is needed, and the
   value of these nodes has no effect on the solution (`run_ic_sensitivity.py`);
3. the obstacle is exactly adiabatic in the discrete sense: solid faces carry
   no flux.

The wall of each component lies on its outermost solid node, so the wall
position is represented to first order, as with any mask.  On the 200×100
grid (`dx = dy = 0.04`) the edges `x = 2.5, 3.5` fall on `h_x` columns and
the edges `y = 1.5, 2.5` on `h_x` rows: `h_x` vanishes exactly on the four
edges (normal to the vertical ones, tangential to the horizontal ones),
while `h_y` vanishes half a cell inside (`figs/fig_obstacle_nodes`).  The
`T` nodes lying on the horizontal edges have a single open face, the `h_y`
node just outside the edge; at steady state the flux through it vanishes,
which places the effective impermeable wall for `h_y` half a cell outside.

## Time integration

Explicit Euler for `T` and `h`; in the full model `Q` is updated first.

    dt = cfl · min( Δ√3/Kn, Δ²/(2Kn²) [, Δ√ε/Kn] ),   Δ = min(dx, dy),  cfl = 0.2

## Verification

`run_verification.py` checks the discretisation on the channel problem
(no obstacle, no-slip walls): the linear model (`Φ = 1`) against the
analytic fully developed profile on grids refined by factors of two, and
the non-linear model with a strong drive against a collocation solution of
the fully developed equation.  `run_selftest.py` checks that the numba
kernel and the numpy reference implementation agree to round-off.
