#=
macro maybe_memoize(type, code)
    if CACHE
        Expr(:escape, :(@memoize $type $code))
    else
        quote
            $(esc(code))
        end
    end
end
=#


macro run_and_gc(code)
    quote
        GC.gc()
        out = $(esc(code))
        GC.gc()
        out
    end
end

function run_and_gc(fun)
    try
        GC.gc()
        fun()
    finally
        GC.gc()
    end
end


"The vector with the memoized functions that should be cleaned when `CACHE[]` is true"
const MEMO_FUNS = []

macro with_memoize(code)
    quote
        out = $(esc(code))
        CACHE[] && foreach(Memoization.empty_cache!, MEMO_FUNS)
        GC.gc()
        out
    end
end
function with_memoize(fun)
    try
        fun()
    finally
        !CACHE[] && foreach(Memoization.empty_cache!, typeof.(MEMO_FUNS))
        GC.gc()
    end
end


const mytime_tabs = Threads.Atomic{Int}(0)
const mytime_tab_str = " | "

"""
Log it with TimerOutputs.
Also, if the verbose variable is defined and set to true, show it
as an info message and end it with the time
"""
macro mytime(msg, code)
    # If not a function definition (just leaving place for an "overloading" to replace the "function" mytime)
    code.head !== :function && return  quote
        @mytime $(Expr(:escape, Expr(:isdefined, :verbose))) && $(Expr(:escape, :verbose)) $(esc(msg)) $(esc(code))
    end
end 

macro mytime(verbose, msg, code)
    quote
        @mytime BEM.to $(esc(verbose)) $(esc(msg)) $(esc(code))
    end
end
macro mytime(to, verbose, msg, code)

    quote
        if $(esc(verbose)) 
            t0 = time_ns()
            mem0 = memused()
            println($mytime_tab_str^mytime_tabs[] * $(esc(msg)) * "...")
            Threads.atomic_add!(BEM.mytime_tabs, 1)
        end
        out = @timeit $(esc(to)) $(esc(msg)) $(esc(code))

        if $(esc(verbose))
            Threads.atomic_add!(BEM.mytime_tabs, -1)
            tottime = round((time_ns() - t0) / 1e9; digits=2)
            mem1 = memused()
            GC.gc()
            mem2 = memused()
            println($mytime_tab_str^mytime_tabs[] * "Done - " * $(esc(msg)) * " in $tottime seconds, using $(memparse(mem1)) (i.e. +$(memparse(mem1 - mem0))) -- after GC, $(memparse(mem2)) (i.e. + $(memparse(mem2 - mem0)))")
            println($mytime_tab_str^mytime_tabs[])
        end

        out
    end
end
function mytime(fun, msg; verbose, gc=false)
    if verbose
        t0 = time_ns()
        mem0 = memused()        
        println(mytime_tab_str^mytime_tabs[] * msg * "...")
        Threads.atomic_add!(BEM.mytime_tabs, 1)
    end
    gc && GC.gc()

    out = @timeit BEM.to msg fun()

    verbose && (mem1 = memused())
    gc && GC.gc()
    if verbose
        Threads.atomic_add!(BEM.mytime_tabs, -1)
        tottime = round((time_ns() - t0) / 1e9; digits=2)
        mem2 = memused()        

        println(mytime_tab_str^mytime_tabs[] * "Done - " * msg * " in $tottime seconds, using $(memparse(mem1)) (i.e. +$(memparse(mem1 - mem0))) -- after GC, $(memparse(mem2)) (i.e. + $(memparse(mem2 - mem0)))")
        println(mytime_tab_str^mytime_tabs[])
    end
    return out
end

mem() = memparse(memused())

function memparse(mbs)
    mbs < 1024 && return "$(round(mbs, digits=2)) MiB"
    return "$(round(mbs / 1024, digits=2)) GiB"
end
memfree() = Meta.parse(match(r"\d+", split(readchomp("/proc/meminfo"), '\n')[2]).match) / 1e3
memtotal() = Meta.parse(match(r"\d+", split(readchomp("/proc/meminfo"), '\n')[1]).match) / 1e3
memused() = memtotal() - memfree()


macro myprogress(code)
    quote
        fun() = $(esc(code))

        if $(Expr(:escape, Expr(:isdefined, :verbose))) && $(Expr(:escape, :verbose)) 
            @progress fun()
        else
            fun()
        end        
    end
end

macro myinfo(msg)
    quote
        printmargin(); print(" ")
        @info $(esc(msg))
        println(mytime_tab_str^mytime_tabs[])
    end
end

printmargin() = print(mytime_tab_str^mytime_tabs[])
function myprintln(msg)
    printmargin(); print(" ")
    println(msg)
end


## TMP aux

macro wait_for_key(msg, code)
    quote
        Base.prompt("Press ENTER for " * $(esc(msg)))
        out = @time $(esc(code))
        @info "Done with " * $(esc(msg))
        out
    end
end