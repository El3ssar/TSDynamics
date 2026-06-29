# DynamicalSystems.jl benchmark worker (the Julia comparison column).
#
# Mirrors benchmarks/runworker.py: reads the shared results/config.json (and the
# dumped from-data series), runs each supported task, times it best-of-N, and
# writes a JSON record in the SAME schema the Python workers emit, so the
# orchestrator merges it transparently. Tasks DynamicalSystems.jl cannot do (or
# whose API errors) are recorded as unsupported/error → blank cells.
#
# Usage: julia --project=julia julia/bench.jl --config results/config.json \
#              --out results/dynamicalsystems_jl.json [--quick] [--only k1,k2]

using DynamicalSystems, OrdinaryDiffEq, StaticArrays, JSON3
using Logging, Statistics
disable_logging(Logging.Warn)

# ----------------------------------------------------------------------------- #
# args / config
# ----------------------------------------------------------------------------- #
getarg(flag) = (i = findfirst(==(flag), ARGS); i === nothing ? nothing : ARGS[i + 1])
cfgpath = getarg("--config")
outpath = getarg("--out")
quick = "--quick" in ARGS
onlyarg = getarg("--only")
only = onlyarg === nothing ? nothing : String.(split(onlyarg, ","))

cfg = JSON3.read(read(cfgpath, String))
L, H, R = cfg.lorenz, cfg.henon, cfg.rossler
LG, INTG, S, NW, REF, SF = cfg.logistic, cfg.integration, cfg.series, cfg.newton, cfg.references, cfg.series_files
fvec(a) = Float64.(collect(a))

readseries(path) = parse.(Float64, readlines(path))

# best-of-N timer (minimum), warm-up paid by the caller before the first timed run
function best_of(f, reps)
    best = Inf
    for _ in 1:reps
        best = min(best, @elapsed f())
    end
    best
end
reps(full, q) = quick ? q : full

# linear-regression slope of y vs x
linslope(x, y) = cov(x, y) / var(x)

# ----------------------------------------------------------------------------- #
# systems
# ----------------------------------------------------------------------------- #
lorenz_rule(u, p, t) = SVector(p[1] * (u[2] - u[1]), u[1] * (p[2] - u[3]) - u[2], u[1] * u[2] - p[3] * u[3])
henon_rule(u, p, n) = SVector(1 - p[1] * u[1]^2 + u[2], p[2] * u[1])
logistic_rule(u, p, n) = SVector(p[1] * u[1] * (1 - u[1]))
rossler_rule(u, p, t) = SVector(-u[2] - u[3], u[1] + p[1] * u[2], p[2] + u[3] * (u[1] - p[3]))
function newton_rule(u, p, n)
    x, y = u[1], u[2]
    a = x^3 - 3x * y^2 - 1; b = 3x^2 * y - y^3
    c = 3x^2 - 3y^2;        d = 6x * y
    den = c^2 + d^2
    SVector(x - (a * c + b * d) / den, y - (b * c - a * d) / den)
end

lorenz_p = [L.params.sigma, L.params.rho, L.params.beta]
lorenz_ic = fvec(L.ic)
henon_p = [H.params.a, H.params.b]; henon_ic = fvec(H.ic)
rossler_p = [R.params.a, R.params.b, R.params.c]; rossler_ic = fvec(R.ic)

make_lorenz(tol) = CoupledODEs(lorenz_rule, lorenz_ic, lorenz_p;
                               diffeq=(alg=Vern9(), abstol=tol, reltol=tol))
make_rossler() = CoupledODEs(rossler_rule, rossler_ic, rossler_p;
                             diffeq=(alg=Vern9(), abstol=1e-9, reltol=1e-9))

# ----------------------------------------------------------------------------- #
# task table: key => (full_reps, quick_reps, () -> (work_closure, estimate_fn))
#   work_closure: zero-arg, does the unit of work (timed)
#   estimate_fn:  () -> Float64 or nothing (the comparable estimate, computed once)
# ----------------------------------------------------------------------------- #
dt = INTG.dt

function task_integrate_short()
    ds = make_lorenz(1e-9)
    T = quick ? 50.0 : INTG.t_short
    (() -> trajectory(ds, T, lorenz_ic; Δt=dt), nothing)
end
function task_integrate_long()
    ds = make_lorenz(1e-9)
    T = quick ? 1000.0 : INTG.t_long
    (() -> trajectory(ds, T, lorenz_ic; Δt=dt), nothing)
end
function task_integrate_accuracy()
    ds = make_lorenz(INTG.acc_rtol)
    T = INTG.t_acc
    ref = fvec(REF.lorenz_acc_final)
    work() = trajectory(ds, T, lorenz_ic; Δt=T)
    est() = (X = first(work()); maximum(abs.(collect(X[end]) .- ref)))
    (work, est)
end
function task_lyapunov_spectrum()
    ds = make_lorenz(1e-9)
    T = quick ? 200.0 : 500.0
    N = round(Int, T / 0.1)
    work() = lyapunovspectrum(ds, N; Δt=0.1, Ttr=20.0)
    (work, () -> maximum(work()))
end
function task_max_lyapunov()
    hm = DeterministicIteratedMap(henon_rule, henon_ic, henon_p)
    N = quick ? 2000 : 5000
    work() = lyapunov(hm, N; Ttr=500)
    (work, work)
end
function task_lyapunov_from_data()
    s = readseries(SF.lorenz_lyap)
    dt_eff = S.dt * S.lyap_stride
    X = embed(s, S.embed_dim, S.lyap_delay)
    ks = 1:25
    work() = lyapunov_from_data(X, ks; w=10)  # Kantz neighbourhood, Theiler w
    function est()
        E = work()
        kt = collect(ks) .* dt_eff
        _, slope = linear_region(kt, E)  # DynamicalSystems.jl's idiomatic auto-fit
        slope
    end
    (work, est)
end
function task_correlation_dimension()
    s = readseries(SF.lorenz_corr)
    X = embed(s, S.embed_dim, S.embed_delay)
    work() = grassberger_proccacia_dim(X)
    (work, work)
end
function task_bifurcation_diagram()
    lm = DeterministicIteratedMap(logistic_rule, [LG.ic], [3.5])
    nr = quick ? 200 : LG.n_rates
    pvals = range(LG.r_min, LG.r_max; length=nr)
    work() = orbitdiagram(lm, 1, 1, pvals; n=LG.n_gens, Ttr=LG.n_discard)
    (work, nothing)
end
function task_basins_of_attraction()
    nm = DeterministicIteratedMap(newton_rule, [0.5, 0.5], Float64[])
    res = quick ? 80 : NW.grid_res
    grid = (range(NW.grid_min, NW.grid_max; length=res), range(NW.grid_min, NW.grid_max; length=res))
    function work()
        mapper = AttractorsViaRecurrences(nm, grid; sparse=true)
        basins_of_attraction(mapper, grid)
    end
    (work, nothing)
end
function task_fixed_points()
    hm = DeterministicIteratedMap(henon_rule, henon_ic, henon_p)
    box = [interval(-2, 2), interval(-2, 2)]  # `interval` re-exported by DynamicalSystems
    work() = fixedpoints(hm, box)
    function est()
        fp, eigs, stab = work()
        maximum(p[1] for p in fp)
    end
    (work, est)
end
function task_poincare_section()
    ros = make_rossler()
    n = quick ? 200 : 1000
    # ~6 time units per revolution → enough total time for n crossings
    work() = poincaresos(ros, (2, 0.0), n * 6.5; direction=+1)
    (work, nothing)
end

TASKS = [
    ("integrate_short", 5, 2, task_integrate_short),
    ("integrate_long", 1, 1, task_integrate_long),
    ("integrate_accuracy", 5, 2, task_integrate_accuracy),
    ("lyapunov_spectrum", 5, 2, task_lyapunov_spectrum),
    ("max_lyapunov", 5, 2, task_max_lyapunov),
    ("lyapunov_from_data", 3, 2, task_lyapunov_from_data),
    ("correlation_dimension", 3, 2, task_correlation_dimension),
    ("bifurcation_diagram", 3, 2, task_bifurcation_diagram),
    ("basins_of_attraction", 1, 1, task_basins_of_attraction),
    ("fixed_points", 5, 2, task_fixed_points),
    ("poincare_section", 3, 2, task_poincare_section),
]

# ----------------------------------------------------------------------------- #
# run
# ----------------------------------------------------------------------------- #
tasks_out = Dict{String,Any}()
for (key, full, q, builder) in TASKS
    if only !== nothing && !(key in only)
        continue
    end
    cell = Dict{String,Any}("status" => "error", "seconds" => nothing,
                            "estimate" => nothing, "note" => nothing)
    try
        work, estimate_fn = builder()
        work()  # warm-up (JIT compile) outside timing
        secs = best_of(work, reps(full, q))
        est = estimate_fn === nothing ? nothing :
              (estimate_fn isa Function ? Float64(estimate_fn()) : nothing)
        cell["status"] = "ok"; cell["seconds"] = secs; cell["estimate"] = est
    catch e
        cell["note"] = sprint(showerror, e)[1:min(end, 200)]
        @info "task failed" key cell["note"]
    end
    tasks_out[key] = cell
    println(stderr, "[julia] $key: $(cell["status"]) ",
            cell["seconds"] === nothing ? "" : "$(round(cell["seconds"] * 1e3; digits=1)) ms")
    flush(stderr)
end

record = Dict(
    "id" => "dynamicalsystems-jl",
    "name" => "DynamicalSystems.jl",
    "language" => "julia",
    "version" => string(pkgversion(DynamicalSystems)),
    "available" => true,
    "reason" => "",
    "tasks" => tasks_out,
)
open(outpath, "w") do io
    JSON3.write(io, record)
end
println(stderr, "wrote $outpath")
