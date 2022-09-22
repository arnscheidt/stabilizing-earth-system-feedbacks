# simple stochastic two-system model 
# Constantin Arnscheidt, 2020

using DifferentialEquations
using Plots
using BenchmarkTools
using DelimitedFiles

# constants
kyr = 1000
Myr = 1000kyr

# FUNCTION SPECIFICATION

# MODEL 1 - single feedback coupled to slow no-feedback system
function model1!(du,u,par,t) 
	du[1] = - u[1]/par[1]
	du[2] = 0
end

function noise1!(du,u,par,t)
	du[1,1] = par[2]
	du[1,2] = 0
	du[2,1] = 0
	du[2,2] = par[3]
end

# MODEL 2 - two feedbacks
function model2!(du,u,par,t) 
	du[1] = - u[1]/par[1]
	du[2] = - u[2]/par[2]
	du[3] = - u[3]/par[3]
	du[4] = 0
end

function noise2!(du,u,par,t)
	du[1,1] = par[4]
	du[2,2] = par[5]
	du[3,3] = par[6]
	du[4,4] = par[7]

end

# MAIN CODE

u0_model1 = [0.0,0.0]
u0_model2 = [0.0,0.0,0.0,0.0] 

tspan = (0.0,200Myr)
tstep = 20 
t = range(tspan[1],tspan[2],step=tstep)

τ₁ = 1kyr
τ₂ = 10kyr
τ₃ = 100kyr

a₁ = 0.03
a₂ = 0.0085
a₃ = 0.0027
a₄ = 0.0015

prob_st = SDEProblem(model2!,noise2!,u0_model2,tspan,[τ₁,τ₂,τ₃,a₁,a₂,a₃,a₄],noise_rate_prototype=zeros(4,4))
sol = solve(prob_st,dt=100,saveat=tstep,EM())
data2 = zeros((length(t),5))
data2[:,1] = t
data2[:,2] = sol[1,:] 
data2[:,3] = sol[2,:] 
data2[:,4] = sol[3,:] 
data2[:,5] = sol[4,:] 

writedlm("feedback_model_output.csv",data2,',')

