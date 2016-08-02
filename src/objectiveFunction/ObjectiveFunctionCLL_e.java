package objectiveFunction;

//import lbfgsb.FunctionValues;
import optimize.FunctionValues;

import EBNC.wdBayes;
import DataStructure.wdBayesNode;
import DataStructure.wdBayesParametersTree;
import Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class ObjectiveFunctionCLL_e extends ObjectiveFunction {

	public ObjectiveFunctionCLL_e(wdBayes algorithm) {
		super(algorithm);
	}

	@Override
	public FunctionValues getValues(double params[]) {

		double negLogLikelihood = 0.0;
		algorithm.dParameters_.copyParameters(params);
		double g[] = new double[algorithm.dParameters_.getNp()];

		int N = algorithm.getNInstances();
		int n = algorithm.getnAttributes();
		int nc = algorithm.getNc();
		double[] myProbs = new double[nc];

		wdBayesParametersTree dParameters = algorithm.getdParameters_();
		Instances instances = algorithm.getM_Instances();		
		wdBayesNode[] myNodes = new wdBayesNode[n];

		int[] order = algorithm.getM_Order();
		double mLogNC = -Math.log(nc); 

		boolean m_Regularization = algorithm.getRegularization();
		double m_Lambda = algorithm.getLambda();

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			wdBayes.findNodesForInstance(myNodes, instance,dParameters);

			// unboxed logDistributionForInstance_d(instance,nodes);
			for (int c = 0; c < nc; c++) {
				myProbs[c] = dParameters.getClassParameter(c);
			}
			for (int u = 0; u < myNodes.length; u++) {
				wdBayesNode bNode = myNodes[u];
				for (int c = 0; c < nc; c++) {
					myProbs[c] += bNode.getXYParameter((int) instance.value(order[u]), c);
				}
			}
			SUtils.normalizeInLogDomain(myProbs);
			negLogLikelihood += (mLogNC - myProbs[x_C]);

			SUtils.exp(myProbs);

			// unboxed logGradientForInstance_d(g, instance,nodes);
			for (int c = 0; c < nc; c++) {
				if (m_Regularization) {
					negLogLikelihood += m_Lambda/2 * dParameters.getClassParameter(c) * dParameters.getClassParameter(c);
					g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) + m_Lambda * dParameters.getClassParameter(c);
				} else {
					g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]);
				}
			}

			for (int u = 0; u < myNodes.length; u++) {
				wdBayesNode bayesNode = myNodes[u];
				for (int c = 0; c < nc; c++) {
					int posp = bayesNode.getXYIndex((int) instance.value(order[u]), c);
					double parameter = bayesNode.getXYParameter((int) instance.value(order[u]), c);

					if (m_Regularization) {
						negLogLikelihood += m_Lambda/2 * parameter * parameter;
						g[posp] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) + m_Lambda * parameter;
					} else {
						g[posp] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]);
					}
				}
			}

			//			// do the eLR trick of normalizing parameters
			//			// for class
			//			for (int c = 0; c < nc; c++) {
			//				double factor = getFactorClass(c, dParameters, nc);
			//				double sum = 0;
			//				for (int k = 0; k < nc; k++) {
			//					sum += (SUtils.ind(k, x_C) - myProbs[k]);
			//				}
			//				g[c] += (-1) * factor * sum;
			//			}

			// do the eLR trick of normalizing parameters
			// for attributes
			for (int u = 0; u < myNodes.length; u++) {
				wdBayesNode bayesNode = myNodes[u];

				for (int k = 0; k < nc; k++) {
					int posp = bayesNode.getXYIndex((int) instance.value(order[u]), k);

					double factor = getClassAttributeFactor(bayesNode, k, u, (int)instance.value(order[u]), dParameters, nc, algorithm.paramsPerAtt[order[u]]);

					double sum = 0;
					for (int v = 0; v < algorithm.paramsPerAtt[order[u]]; v++) {
						sum += SUtils.ind((int) instance.value(order[u]), v) * (SUtils.ind(k, x_C) - myProbs[k]);						
					}

					g[posp] += (-1) * factor * sum;
				}
			}
		}
		
		if (Double.isNaN(negLogLikelihood )) {
			System.err.println("Objective function is NaN, your parameters have gone rougue, please check.");
		}

		return new FunctionValues(negLogLikelihood, g);
	}

	private double getClassAttributeFactor(wdBayesNode bayesNode, int k, int u, int x_u, wdBayesParametersTree dParameters, int nc, int paramsPerAtt) {

		double classAttProb = 0;
		double parameter = bayesNode.getXYParameter(x_u, k);

		classAttProb = Math.exp(parameter);
		
		if (Double.isInfinite(classAttProb)) {
			return 1.0;
		}

		double sum = 0;
		for (int v = 0; v < paramsPerAtt; v++) {
			sum += Math.exp(bayesNode.getXYParameter(v, k));
		}
		
		if (Double.isInfinite(sum)) {
			return 1.0;
		}

		double factor = classAttProb / sum;
		return factor;		
	}

	private double getFactorClass(int c, wdBayesParametersTree dParameters, int nc) {
		double classProb = 0;
		classProb = Math.exp(dParameters.getClassParameter(c)); 

		double sum = 0;
		for (int k = 0; k < nc; k++) {
			sum += Math.exp(dParameters.getClassParameter(k));
		}

		double factor = classProb/sum;
		return factor;
	}


}
