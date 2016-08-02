package Utils;

import DataStructure.xxyDist;
import DataStructure.xyDist;

public class CorrelationMeasures {

	/**
	 * Find maximum spanning tree
	 * 
	 * @param numNode: Number of node in the graph
	 * @param weight: weight[i][j] = weight[j][i] is the weight of the edge from node i to node j
	 * @exception Exception: if the instance could not be incorporated in the model.
	 */
	public static void findMST(int numNode, double[][] weight, int[] m_Parents) {

		// included in tree? If yes, this value is 1 otherwise is -1
		int[] inTree = new int[numNode];
		// distance[i] is the largest weight of node i to those nodes already in
		// the tree
		double[] distance = new double[numNode];

		for (int i = 0; i < numNode; i++) {
			inTree[i] = -1;
		}
		// Add node 0 to the tree
		inTree[0] = 1;
		// set distances of other nodes
		for (int i = 1; i < numNode; i++) {
			distance[i] = weight[0][i];
			m_Parents[i] = 0;
		}

		for (int treeSize = 1; treeSize < numNode; treeSize++) {
			// Find the node with the largest distance to the tree
			int max = -1;
			for (int i = 0; i < numNode; i++) {
				// consider nodes that are not in the tree
				if (inTree[i] != 1) {
					if ((max == -1) || (distance[max] < distance[i])) {
						max = i;
					}
				}
			}

			// add node max to the tree
			inTree[max] = 1;
			// update distances for nodes that are not in the tree as new node
			// max is added to the tree
			for (int i = 0; i < numNode; i++) {
				if (inTree[i] == 1)
					continue;
				if (distance[i] < weight[max][i]) {
					distance[i] = weight[max][i];
					m_Parents[i] = max;
				}
			}
		}

	}

	public static int[] getMIbasedParent(xxyDist xxyDist_) throws Exception {

		int nc = xxyDist_.getNoClasses();
		int n = xxyDist_.getNoAtts();

		int[] paramsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			paramsPerAtt[u] = xxyDist_.getNoValues(u);		
		}

		int[] m_Parents = new int[n];
		for (int u = 0; u < n; u++) {
			m_Parents[u] = -1;
		}

		double[][] m_CondiMutualInfo = new double[n][n];

		// Calculate conditional mutual information		
		for (int u1 = 0; u1 < n; u1++) {
			for (int u2 = u1; u2 >= 0; u2--) {
				if (u1 == u2) 
					continue;

				m_CondiMutualInfo[u1][u2] = 0;

				for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {					
					for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {					

						for (int c = 0; c < nc; c++) {							

							double mi = 0;							
							//mi = (m_2vCondiCounts[posNN] / N) * Math.log((m_2vCondiCounts[posNN] * m_ClassCounts[c]) / (m_1vCondiCounts[pos1] * m_1vCondiCounts[pos2]));
							mi = xxyDist_.jointP(u1, u1val, u2, u2val, c) * 
									Math.log((xxyDist_.jointP(u1, u1val, u2, u2val, c) * xxyDist_.xyDist_.p(c)) / 
											(xxyDist_.xyDist_.jointP(u1, u1val, c) * xxyDist_.xyDist_.jointP(u2, u2val, c)));

							// Compute conditional mutual information
							m_CondiMutualInfo[u1][u2] += mi;
							m_CondiMutualInfo[u2][u1] += mi;
						}
					}
				}
			}
		}				

		double[] MI = new double[n];
		for (int att1 = 0; att1 < n; att1++) {
			for (int att2 = 0; att2 < n; att2++) {
				MI[att1] += m_CondiMutualInfo[att1][att2];
			}
		}

		// Find maximum spanning tree
		CorrelationMeasures.findMST(n, m_CondiMutualInfo, m_Parents);
		
		return m_Parents;
	}

	/**
	 *
	 *             __
	 *             \               P(x,y)
	 *  MI(X,Y)=   /_  P(x,y)log------------
	 *            x,y             P(x)P(y)
	 *
	 *
	 */
	public static void getMutualInformation(xyDist xyDist_, double[] mi) {
		int nc = xyDist_.getNoClasses();
		int n = xyDist_.getNoAtts();
		double N = xyDist_.getNoData();

		int[] paramsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			paramsPerAtt[u] = xyDist_.getNoValues(u);		
		}

		for (int u = 0; u < n; u++) {
			double m = 0;
			for (int uval = 0; uval < paramsPerAtt[u]; uval++) {
				for (int y = 0; y < nc; y++) {
					int avyCount = xyDist_.getCount(u, uval, y);
					if (avyCount > 0) {
						m += (avyCount / N) * Math.log( avyCount / ( xyDist_.getCount(u, uval)/N * xyDist_.getClassCount(y) ) ) / Math.log(2);
						//System.out.println(avyCount + ", " + xyDist_.getCount(u, uval) + ", " + xyDist_.getClassCount(y) + " = " + m);
					}
				}
			}
			mi[u] = m;
		}
	}
	

	/**
	 *                 __
	 *                 \                    P(x1,x2|y)
	 * CMI(X1,X2|Y)= = /_   P(x1,x2,y) log-------------
	 *               x1,x2,y              P(x1|y)P(x2|y)
	 *
	 */
	public static void getCondMutualInf(xxyDist xxyDist_, double[][] cmi) {	

		int nc = xxyDist_.getNoClasses();
		int n = xxyDist_.getNoAtts();
		double N = xxyDist_.getNoData();

		int[] paramsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			paramsPerAtt[u] = xxyDist_.getNoValues(u);		
		}

		// Calculate conditional mutual information		
		for (int u1 = 1; u1 < n; u1++) {
			for (int u2 = 0; u2 < u1; u2++) {
				
				double mi = 0;
				for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {					
					for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
						for (int c = 0; c < nc; c++) {		

							double avvyCount = xxyDist_.getCount(u1, u1val, u2, u2val, c);
							if (avvyCount > 0) {															
								//mi += xxyDist_.jointP(u1, u1val, u2, u2val, c) * Math.log((xxyDist_.jointP(u1, u1val, u2, u2val, c) * xxyDist_.xyDist_.p(c)) / (xxyDist_.xyDist_.jointP(u1, u1val, c) * xxyDist_.xyDist_.jointP(u2, u2val, c)));
								//mi += (avvyCount/N) * (Math.log((avvyCount * xxyDist_.xyDist_.getClassCount(c)) / (xxyDist_.xyDist_.getCount(u1, u1val, c) * xxyDist_.xyDist_.getCount(u2, u2val, c))) / Math.log(2));
								
								double a = avvyCount;
								double b = xxyDist_.xyDist_.getClassCount(c);
								double d = xxyDist_.xyDist_.getCount(u1, u1val, c);
								double e = xxyDist_.xyDist_.getCount(u2, u2val, c);
								
								//System.out.println((a/N) * Math.log((a*b) / (d*e)) / Math.log(2));
								double mitemp = (a/N) * Math.log((a*b) / (d*e)) / Math.log(2);
								mi += mitemp;
								
								//double mitemp2 = ((avvyCount/N) * Math.log( (avvyCount * xxyDist_.xyDist_.getClassCount(c)) / (xxyDist_.xyDist_.getCount(u1, u1val, c) * xxyDist_.xyDist_.getCount(u2, u2val, c)) ) / Math.log(2));
								//mi += mitemp2;
								
								//mi += ((avvyCount/N) * Math.log( (avvyCount * xxyDist_.xyDist_.getClassCount(c)) / (xxyDist_.xyDist_.getCount(u1, u1val, c) * xxyDist_.xyDist_.getCount(u2, u2val, c)) ) / Math.log(2));								
								//System.out.println((avvyCount/N) * Math.log( (avvyCount * xxyDist_.xyDist_.getClassCount(c)) / (xxyDist_.xyDist_.getCount(u1, u1val, c) * xxyDist_.xyDist_.getCount(u2, u2val, c)) ) / Math.log(2));
								
								//System.out.println(avvyCount + ", " + xxyDist_.xyDist_.getClassCount(c) + ", " + xxyDist_.xyDist_.getCount(u1, u1val, c) + ", " + xxyDist_.xyDist_.getCount(u2, u2val, c) + ": " + mi);												
								
							}
						}
					}
				}
				cmi[u1][u2] = mi;
				cmi[u2][u1] = mi;
			}
		}		

	}


}
