package objectiveFunction.parallel;

import java.util.Arrays;
import java.util.concurrent.Callable;

import EBNC.wdBayes;
import DataStructure.wdBayesNode;
import DataStructure.wdBayesParametersTree;
import Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class CallableCLL_e implements Callable<Double> {

	private Instances instances;
	private int start;
	private int stop;
	wdBayesNode[] myNodes;
	private double[] myProbs;
	private wdBayesParametersTree dParameters;
	private int[] order;
	private int nc;
	private double[] g;
	private double mLogNC;
	private wdBayes algorithm;

	public CallableCLL_e(Instances instances, int start, int stop, int nc,wdBayesNode[] nodes, double[] myProbs, double[]g,wdBayesParametersTree dParameters, int[] order, wdBayes algorithm) {
		this.algorithm = algorithm;
		this.instances = instances;
		this.start = start;
		this.stop = stop;
		this.nc= nc;
		this.myNodes = nodes;
		this.myProbs = myProbs;
		this.g = g;
		this.dParameters = dParameters;
		this.order = order;
		this.mLogNC = -Math.log(nc); 
	}

	@Override
	public Double call() throws Exception {

		boolean m_Regularization = algorithm.getRegularization();
		double m_Lambda = algorithm.getLambda();

		double negLogLikelihood = 0.0;

		Arrays.fill(g, 0.0);

		for (int i = start; i <= stop; i++) {
			Instance instance = instances.instance(i);
			int x_C = (int) instance.classValue();

			wdBayes.findNodesForInstance(myNodes, instance, dParameters);
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
		return negLogLikelihood;
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
