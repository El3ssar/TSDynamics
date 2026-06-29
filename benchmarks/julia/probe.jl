# Empirical API probe for DynamicalSystems.jl v3 — discover exact signatures.
using DynamicalSystems, OrdinaryDiffEq, StaticArrays
using Logging
disable_logging(Logging.Warn)

function try_it(name, f)
    try
        v = f()
        println("OK  ", name, "  => ", v); flush(stdout)
    catch e
        println("ERR ", name, "  => ", sprint(showerror, e)[1:min(end,160)]); flush(stdout)
    end
end

# --- systems ---
lorenz_rule(u,p,t) = SVector(p[1]*(u[2]-u[1]), u[1]*(p[2]-u[3])-u[2], u[1]*u[2]-p[3]*u[3])
henon_rule(u,p,n)  = SVector(1 - p[1]*u[1]^2 + u[2], p[2]*u[1])
logistic_rule(u,p,n) = SVector(p[1]*u[1]*(1-u[1]))
rossler_rule(u,p,t) = SVector(-u[2]-u[3], u[1]+p[1]*u[2], p[2]+u[3]*(u[1]-p[3]))

ode = CoupledODEs(lorenz_rule, [1.0,1.0,1.0], [10.0,28.0,8/3]; diffeq=(alg=Vern9(), abstol=1e-9, reltol=1e-9))
hm  = DeterministicIteratedMap(henon_rule, [0.1,0.1], [1.4,0.3])
lm  = DeterministicIteratedMap(logistic_rule, [0.5], [3.5])

try_it("trajectory Δt", () -> size(first(trajectory(ode, 10.0; Δt=0.01))))
try_it("lyapunovspectrum", () -> lyapunovspectrum(ode, 1000; Δt=0.05, Ttr=20.0))
try_it("lyapunov ode", () -> lyapunov(ode, 500.0; Ttr=20.0))
try_it("lyapunov henon", () -> lyapunov(hm, 5000))

X = first(trajectory(ode, 200.0; Δt=0.05, Ttr=20.0))
try_it("grassberger_proccacia_dim(X)", () -> grassberger_proccacia_dim(X))
try_it("correlationsum exists", () -> (correlationsum, true)[2])
try_it("generalized_dim", () -> generalized_dim(X))

# from-data lyapunov
try_it("lyapunov_from_data", () -> lyapunov_from_data(X, 1:20))

# orbit diagram
try_it("orbitdiagram", () -> length(orbitdiagram(lm, 1, 1, range(2.8,4.0;length=50); n=50, Ttr=100)))

# fixed points
try_it("fixedpoints", () -> begin
    box = interval(-2,2) × interval(-2,2)
    fp, eigs, stab = fixedpoints(hm, box)
    fp
end)

# poincare
ros = CoupledODEs(rossler_rule, [1.0,1.0,1.0], [0.2,0.2,5.7]; diffeq=(alg=Vern9(), abstol=1e-9, reltol=1e-9))
try_it("poincaresos", () -> size(poincaresos(ros, (2, 0.0), 2000.0)))
try_it("PoincareMap", () -> begin
    pmap = PoincareMap(ros, (2, 0.0))
    size(first(trajectory(pmap, 500)))
end)

# basins
newton_rule(u,p,n) = begin
    x,y = u[1], u[2]
    a = x^3 - 3x*y^2 - 1; b = 3x^2*y - y^3
    c = 3x^2 - 3y^2;      d = 6x*y
    den = c^2 + d^2
    SVector(x - (a*c + b*d)/den, y - (b*c - a*d)/den)
end
nm = DeterministicIteratedMap(newton_rule, [0.5,0.5], Float64[])
try_it("AttractorsViaRecurrences+basins", () -> begin
    grid = (range(-1.5,1.5;length=50), range(-1.5,1.5;length=50))
    mapper = AttractorsViaRecurrences(nm, grid; sparse=true)
    basins, atts = basins_of_attraction(mapper, grid)
    (length(atts), size(basins))
end)
println("PROBE DONE")
