package objectiveFunction.parallel;

import java.util.Arrays;
import java.util.concurrent.Callable;

import EBNC.wdBayes;
import DataStructure.wdBayesNode;
import DataStructure.wdBayesParametersTree;
import Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class CallableCLL_d implements Callable<Double>{

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

	public CallableCLL_d(Instances instances, int start, int stop, int nc,wdBayesNode[] nodes, double[] myProbs, double[]g,wdBayesParametersTree dParameters, int[] order, wdBayes algorithm) {
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
		boolean m_TowardsNaiveBayes = algorithm.getTowardsNaiveBayes();
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
			//negLogLikelihood += (- myProbs[x_C]);
			SUtils.exp(myProbs);

			// unboxed logGradientForInstance_d(g, instance,nodes);
			for (int c = 0; c < nc; c++) {
				if (m_Regularization) {
					if (m_TowardsNaiveBayes) {
						double reg = dParameters.getClassParameter(c) - dParameters.getClassCounts()[c];
						negLogLikelihood += m_Lambda/2 * Math.pow(reg, 2); 
						g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c])  + m_Lambda * reg;
					} else {
						negLogLikelihood += m_Lambda/2 * dParameters.getClassParameter(c) * dParameters.getClassParameter(c);
						g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) + m_Lambda * dParameters.getClassParameter(c);
					}
				} else {
					g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]);
				}
			}

			for (int u = 0; u < myNodes.length; u++) {
				wdBayesNode bayesNode = myNodes[u];
				for (int c = 0; c < nc; c++) {
					int posp = bayesNode.getXYIndex((int) instance.value(order[u]), c);
					double parameter = bayesNode.getXYParameter((int) instance.value(order[u]), c);
					double probability = bayesNode.getXYCount((int) instance.value(order[u]), c);
					
					if (m_Regularization) {
						if (m_TowardsNaiveBayes) { 
							double reg = parameter - probability;
							negLogLikelihood += m_Lambda/2 * reg * reg;
							g[posp] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) + m_Lambda * reg;
						} else {
							negLogLikelihood += m_Lambda/2 * parameter * parameter;
							g[posp] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) + m_Lambda * parameter;
						}
					} else {
						g[posp] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]);
					}
				}
			}			

		}
		return negLogLikelihood;
	}

}