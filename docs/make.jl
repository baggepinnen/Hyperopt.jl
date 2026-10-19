ENV["GKSwstype"] = "100" # The GR backend of Plots renders without a display

using Documenter
using Hyperopt
using Optim # Loads the extension that defines `hyperoptim`
using Plots

# `checkdocs = :exports` detects docstrings of exported names that are not included in the manual, but not exported
# names without a docstring. `Docs.undocumented_names` is available on Julia 1.11 and later.
if isdefined(Base.Docs, :undocumented_names)
    undocumented = Base.Docs.undocumented_names(Hyperopt)
    isempty(undocumented) || error("The following exported names have no docstring: $(join(undocumented, ", "))")
end

makedocs(
    sitename = "Hyperopt.jl",
    authors = "Fredrik Bagge Carlson",
    modules = [Hyperopt],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://baggepinnen.github.io/Hyperopt.jl/stable/",
        edit_link = "master",
    ),
    pages = [
        "Home" => "index.md",
        "The `@hyperopt` macro" => "hyperopt.md",
        "Samplers" => "samplers.md",
        "Hyperband" => "hyperband.md",
        "BOHB" => "bohb.md",
        "Function interface" => "function_interface.md",
        "Optim" => "optim.md",
        "Plotting" => "plotting.md",
        "API" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/baggepinnen/Hyperopt.jl.git",
    devbranch = "master",
)
