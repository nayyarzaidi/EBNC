package objectiveFunction;

import EBNC.wdBayes;

//import lbfgsb.DifferentiableFunction;
//import lbfgsb.FunctionValues;

import optimize.DifferentiableFunction;
import optimize.FunctionValues;

public abstract class ObjectiveFunction implements DifferentiableFunction {

	protected final wdBayes algorithm;
	
	public ObjectiveFunction(wdBayes algorithm) {
		this.algorithm = algorithm;
	}

	@Override
	abstract public FunctionValues getValues(double params[]);	
	
	public void finish(){
		
	}
	
}