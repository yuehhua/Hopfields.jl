using Hopfields
using Documenter

DocMeta.setdocmeta!(Hopfields, :DocTestSetup, :(using Hopfields); recursive=true)

makedocs(;
    modules=[Hopfields],
    authors="Yueh-Hua Tu",
    repo="https://github.com/yuehhua/Hopfields.jl/blob/{commit}{path}#{line}",
    sitename="Hopfields.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://yuehhua.github.io/Hopfields.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/yuehhua/Hopfields.jl",
    devbranch="main",
)
