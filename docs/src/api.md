# API

```@docs
Hyperopt
```

## Macros

```@docs
@hyperopt
@phyperopt
@thyperopt
```

## Parallel evaluation

```@docs
Hyperopt.tmap
```

## Hyperoptimizer

```@docs
Hyperoptimizer
Base.iterate(::Hyperoptimizer)
Base.minimum(::Hyperoptimizer)
Base.maximum(::Hyperoptimizer)
printmin
printmax
warn_on_boundary
```

## Sampler types

```@docs
RandomSampler
LHSampler
CLHSampler
Hyperband
BOHB
```

## Dimension types

```@docs
Continuous
Categorical
UnorderedCategorical
Hyperopt.DimensionType
```

## Hyperband functions

```@docs
hyperband
Hyperopt.successive_halving
```

## Optim extension

```@docs
hyperoptim
```

## Settings

```@docs
Hyperopt.SHOW_PROGRESS
```
