package objectiveFunction.parallel;

import java.util.Arrays;
import java.util.concurrent.Callable;

import EBNC.wdBayes;
import DataStructure.wdBayesNode;
import DataStructure.wdBayesParametersTree;
import Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class CallableMSE_w implements Callable<Double>{

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

	public CallableMSE_w(Instances instances, int start, int stop, int nc,wdBayesNode[] nodes, double[] myProbs, double[]g,wdBayesParametersTree dParameters, int[] order, wdBayes algorithm) {
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
		
		double meanSquareError = 0.0;
		Arrays.fill(g, 0.0);
		
		for (int i = start; i <= stop; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			wdBayes.findNodesForInstance(myNodes, instance, dParameters);

			// unboxed logDistributionForInstance_w
			for (int c = 0; c < nc; c++) {
				myProbs[c] = dParameters.getClassCounts()[c] * dParameters.getClassParameter(c);
			}
			
			for (int c = 0; c < nc; c++) {
				for (int u = 0; u < myNodes.length; u++) {
					wdBayesNode bNode = myNodes[u];
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
		
		return meanSquareError;
	}
	
}