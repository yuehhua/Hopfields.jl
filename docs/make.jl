using scTransformer
using Documenter

DocMeta.setdocmeta!(scTransformer, :DocTestSetup, :(using scTransformer); recursive=true)

makedocs(;
    modules=[scTransformer],
    authors="Yueh-Hua Tu",
    repo="https://github.com/yuehhua/scTransformer.jl/blob/{commit}{path}#{line}",
    sitename="scTransformer.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://yuehhua.github.io/scTransformer.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/yuehhua/scTransformer.jl",
    devbranch="main",
)
