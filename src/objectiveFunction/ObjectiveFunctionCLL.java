package objectiveFunction;

import EBNC.wdBayes;

//import lbfgsb.DifferentiableFunction;
//import lbfgsb.FunctionValues;

import optimize.DifferentiableFunction;
import optimize.FunctionValues;

public abstract class ObjectiveFunctionCLL implements DifferentiableFunction {

	protected final wdBayes algorithm;
	
	public ObjectiveFunctionCLL(wdBayes algorithm) {
		this.algorithm = algorithm;
	}

	@Override
	abstract public FunctionValues getValues(double params[]);	
	
	public void finish(){
		
	}
	
}