package objectiveFunction;

//import lbfgsb.FunctionValues;
import optimize.FunctionValues;

import EBNC.wdBayes;
import DataStructure.wdBayesNode;
import DataStructure.wdBayesParametersTree;
import Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class ObjectiveFunctionMSE_w extends ObjectiveFunction {

	public ObjectiveFunctionMSE_w(wdBayes algorithm) {
		super(algorithm);
	}

	@Override
	public FunctionValues getValues(double params[]) {

		double meanSquareError = 0.0;
		int nc = algorithm.getNc();

		// Copy params into XYParameters
		algorithm.dParameters_.copyParameters(params);

		double g[] = new double[algorithm.dParameters_.getNp()];

		int N = algorithm.getNInstances();
		int n = algorithm.getnAttributes();
		//int nc = algorithm.getNc();
		double[] myProbs = new double[nc];

		wdBayesParametersTree dParameters = algorithm.getdParameters_();
		Instances instances = algorithm.getM_Instances();
		wdBayesNode[] myNodes = new wdBayesNode[n];

		int[] order = algorithm.getM_Order();
		
		boolean m_Regularization = algorithm.getRegularization();
		double m_Lambda = algorithm.getLambda();

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);
			int x_C = (int) instance.classValue();

			wdBayes.findNodesForInstance(myNodes, instance, dParameters);

			// unboxed logDistributionForInstance_w
			for (int c = 0; c < nc; c++) {
				//System.out.println(xyDist.pp(c)  + " , " + dParameters.getClassParameter(c) + " = " + xyDist.pp(c) * dParameters.getClassParameter(c));
				myProbs[c] = dParameters.getClassCounts()[c] * dParameters.getClassParameter(c);
			}

			for (int c = 0; c < nc; c++) {
				for (int u = 0; u < myNodes.length; u++) {
					wdBayesNode bNode = myNodes[u];
					//System.out.println(bNode.getXYParameter((int) instance.value(order[u]), c) + " , " +  bNode.getXYCount((int) instance.value(order[u]), c) + " = " + bNode.getXYParameter((int) instance.value(order[u]), c) * bNode.getXYCount((int) instance.value(order[u]), c));
					myProbs[c] += bNode.getXYParameter((int) instance.value(order[u]), c) * bNode.getXYCount((int) instance.value(order[u]), c);
				}
			}

			SUtils.normalizeInLogDomain(myProbs);
			SUtils.exp(myProbs);
			
			double prod = 0;
			for (int y = 0; y < nc; y++) {
				prod += (SUtils.ind(y, x_C) - myProbs[y]) * (SUtils.ind(y, x_C) - myProbs[y]);
				//meanSquareError += (prod * prod);
			}
			//meanSquareError += prod/nc;
			meanSquareError += prod;

			// unboxed logGradientForInstance_w
			for (int c = 0; c < nc; c++) {
				if (m_Regularization) {
					meanSquareError += m_Lambda/2 * dParameters.getClassParameter(c) * dParameters.getClassParameter(c);
					//g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * xyDist.pp(c)  + m_Lambda * dParameters.getClassParameter(c);
					for (int k = 0; k < nc; k++) {
						g[c] += (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c] * dParameters.getClassCounts()[c] + m_Lambda * dParameters.getClassParameter(c);
					}
				} else {
					//g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * xyDist.pp(c);
					for (int k = 0; k < nc; k++) {
						g[c] += (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c] * dParameters.getClassCounts()[c];
					}
				}
			}

			for (int u = 0; u < myNodes.length; u++) {
				wdBayesNode bayesNode = myNodes[u];
				for (int c = 0; c < nc; c++) {
					int index = bayesNode.getXYIndex((int) instance.value(order[u]), c);
					double probability = bayesNode.getXYCount((int) instance.value(order[u]), c);
					double parameter = bayesNode.getXYParameter((int) instance.value(order[u]), c);

					if (m_Regularization) {
						meanSquareError += m_Lambda/2 * parameter * parameter;
						//g[index] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * probability + m_Lambda * parameter;
						for (int k = 0; k < nc; k++) {
							g[index] += (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c] * probability + m_Lambda * parameter;
						}
					} else {
						//g[index] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * probability;
						for (int k = 0; k < nc; k++) {
							g[index] += (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c] * probability;
						}
					}
				}
			}
		}
		
		FunctionValues fv = new FunctionValues(meanSquareError, g);
		//System.out.println("fv="+fv.functionValue+"\t"+negLogLikelihood);
		//System.out.print(negLogLikelihood);
		return fv;
	}

}