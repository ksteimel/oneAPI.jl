using LinearAlgebra
import Adapt

@testset "constructors" begin
  xs = oneArray{Int}(undef, 2, 3)
  @test collect(oneArray([1 2; 3 4])) == [1 2; 3 4]
  @test testf(vec, rand(5,3))
  @test Base.elsize(xs) == sizeof(Int)
  @test oneArray{Int, 2}(xs) === xs

  @test_throws ArgumentError Base.unsafe_convert(Ptr{Int}, xs)
  @test_throws ArgumentError Base.unsafe_convert(Ptr{Float32}, xs)

  @test collect(oneAPI.zeros(2, 2)) == zeros(2, 2)
  @test collect(oneAPI.ones(2, 2)) == ones(2, 2)

  @test collect(oneAPI.fill(0, 2, 2)) == zeros(2, 2)
  @test collect(oneAPI.fill(1, 2, 2)) == ones(2, 2)
end

@testset "adapt" begin
  A = rand(Float32, 3, 3)
  dA = oneArray(A)
  @test Adapt.adapt(Array, dA) == A
  @test Adapt.adapt(oneArray, A) isa oneArray
  @test Array(Adapt.adapt(oneArray, A)) == A
end

@testset "oapi" begin
  cpu_64bit = rand(Float64, 3,3)
  cpu_32bit = convert(Matrix{Float32}, cpu_64bit)
  oapi_32bit = oneArray(cpu_32bit)
  oapi_mult = oapi(*)
  # Test that 32 bit floats are the same as passing through oneArray
  @test isequal(oapi(cpu_32bit), oapi_32bit)
  # Test that 64 bit floats are converted to 32 bit
  @test isequal(oapi(cpu_64bit), oapi_32bit)
  # Test that functions are passed through unmodified
  @test oapi_mult(oapi_32bit, oapi_32bit) == oapi_32bit * oapi_32bit
end

@testset "reshape" begin
  A = [1 2 3 4
       5 6 7 8]
  gA = reshape(oneArray(A),1,8)
  _A = reshape(A,1,8)
  _gA = Array(gA)
  @test all(_A .== _gA)
  A = [1,2,3,4]
  gA = reshape(oneArray(A),4)
end

@testset "fill(::SubArray)" begin
  xs = oneAPI.zeros(Float32, 3)
  fill!(view(xs, 2:2), 1)
  @test Array(xs) == [0,1,0]
end
