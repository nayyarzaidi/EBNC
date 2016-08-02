package DataStructure;

import java.util.Arrays;

import Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class wdBayesParametersTree {

	private double[] parameters;
	private int np;

	private wdBayesNode[] wdBayesNode_;
	private int[] activeNumNodes;

	private int N;
	private int n;
	private int nc;

	private int[] m_ParamsPerAtt;

	private int[] order;
	private int[][] parents;

	private int scheme;

	private double[] classCounts;
	private double[] classBuffer;

	double[]  mi;
	double[][] fmi;

	/**
	 * Constructor called by wdBayes
	 */
	public wdBayesParametersTree(int n, int nc, int[] paramsPerAtt, int[] m_Order, int[][] m_Parents, int m_P) {
		this.n = n;
		this.nc = nc;
		scheme = m_P;

		m_ParamsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			m_ParamsPerAtt[u] = paramsPerAtt[u];
		}

		order = new int[n];
		parents = new int[n][];

		for (int u = 0; u < n; u++) {
			order[u] = m_Order[u];
		}

		activeNumNodes = new int[n];		

		for (int u = 0; u < n; u++) {
			if (m_Parents[u] != null) {
				parents[u] = new int[m_Parents[u].length];
				for (int p = 0; p < m_Parents[u].length; p++) {
					parents[u][p] = m_Parents[u][p];
				}
			}
		}

		wdBayesNode_ = new wdBayesNode[n];
		for (int u = 0; u < n; u++) {
			wdBayesNode_[u] = new wdBayesNode(scheme);
			wdBayesNode_[u].init(nc, paramsPerAtt[m_Order[u]]);
			//wdBayesNode_[u].init(nc, paramsPerAtt[u]);
		}

		classCounts = new double[nc];
		classBuffer = new double[nc];

		mi = new double[n];
		
		fmi = new double[n][];
		for (int u = 0; u < n; u++) {
			fmi[u] = new double[paramsPerAtt[u]];
		}
		
	}

	/* 
	 * -----------------------------------------------------------------------------------------
	 * Update count statistics that is:  relevant ***xyCount*** in every node
	 * -----------------------------------------------------------------------------------------
	 */

	public void update(Instance instance) {
		classCounts[(int) instance.classValue()]++;

		for (int u = 0; u < n; u++) {
			updateAttributeTrie(instance, u, order[u], parents[u]);
		}

		N++;
	}

	public void updateAttributeTrie(Instance instance, int i, int u, int[] lparents) {

		int x_C = (int) instance.classValue();
		int x_u = (int) instance.value(u);		

		wdBayesNode_[i].incrementXYCount(x_u, x_C);	

		if (lparents != null) {

			wdBayesNode currentdtNode_ = wdBayesNode_[i];

			for (int d = 0; d < lparents.length; d++) {
				int p = lparents[d];

				if (currentdtNode_.att == -1 || currentdtNode_.children == null) {
					currentdtNode_.children = new wdBayesNode[m_ParamsPerAtt[p]];
					currentdtNode_.att = p;	
				}

				int x_up = (int) instance.value(p);
				currentdtNode_.att = p;

				// the child has not yet been allocated, so allocate it
				if (currentdtNode_.children[x_up] == null) {
					currentdtNode_.children[x_up] = new wdBayesNode(scheme);
					currentdtNode_.children[x_up].init(nc, m_ParamsPerAtt[u]);
				} 

				currentdtNode_.children[x_up].incrementXYCount(x_u, x_C);
				currentdtNode_ = currentdtNode_.children[x_up];
			}
		}
	}

	/* 
	 * -----------------------------------------------------------------------------------------
	 * Convert count into (NB) probabilities
	 * -----------------------------------------------------------------------------------------
	 */

	public void countsToProbability() {
		for (int c = 0; c < nc; c++) {
			classCounts[c] = (SUtils.MEsti(classCounts[c], N, nc));
		}
		for (int u = 0; u < n; u++) {
			convertCountToProbs(order[u], parents[u], wdBayesNode_[u]);
		}
	}

	public void convertCountToProbs(int u, int[] lparents, wdBayesNode pt) {

		int att = pt.att;

		if (att == -1) {
			int[][] tempArray = new int[m_ParamsPerAtt[u]][nc];
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					tempArray[uval][y] = (int) pt.getXYCount(uval, y);
				}
			}
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					int denom = 0;
					for (int dval = 0; dval < m_ParamsPerAtt[u]; dval++) {
						denom += tempArray[dval][y];
					}
					double prob = Math.log(Math.max(SUtils.MEsti(tempArray[uval][y], denom, m_ParamsPerAtt[u]),1e-75));
					pt.setXYCount(uval, y, prob);
				}
			}			
			return;
		}

		while (att != -1) {
			/*
			 * Now convert non-leaf node counts to probs
			 */
			int[][] tempArray = new int[m_ParamsPerAtt[u]][nc];
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					tempArray[uval][y] = (int) pt.getXYCount(uval, y);
				}
			}
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					int denom = 0;
					for (int dval = 0; dval < m_ParamsPerAtt[u]; dval++) {
						denom += tempArray[dval][y];
					}
					double prob = Math.log(Math.max(SUtils.MEsti(tempArray[uval][y], denom, m_ParamsPerAtt[u]),1e-75));
					pt.setXYCount(uval, y, prob);
				}
			}

			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null) 					
					convertCountToProbs(u, lparents, next);

				// Flag end of nodes
				att = -1;				
			}			
		}

		return;
	}

	/* 
	 * -----------------------------------------------------------------------------------------
	 * Convert count into (Uniform) probabilities
	 * -----------------------------------------------------------------------------------------
	 */

	public void countsToUniformProbability() {
		for (int c = 0; c < nc; c++) {
			classCounts[c] = Math.log(SUtils.MEsti(classCounts[c], N, nc));
		}
		for (int u = 0; u < n; u++) {
			convertCountToUniformProbs(order[u], parents[u], wdBayesNode_[u]);
		}
	}

	public void convertCountToUniformProbs(int u, int[] lparents, wdBayesNode pt) {

		int att = pt.att;

		if (att == -1) {
			int[][] tempArray = new int[m_ParamsPerAtt[u]][nc];
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					tempArray[uval][y] = (int) pt.getXYCount(uval, y);
				}
			}
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					int denom = 0;
					for (int dval = 0; dval < m_ParamsPerAtt[u]; dval++) {
						denom += tempArray[dval][y];
					}
					double prob = Math.log(Math.max(1/m_ParamsPerAtt[u],1e-75));
					pt.setXYCount(uval, y, prob);
				}
			}			
			return;
		}

		while (att != -1) {
			/*
			 * Now convert non-leaf node counts to probs
			 */
			int[][] tempArray = new int[m_ParamsPerAtt[u]][nc];
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					tempArray[uval][y] = (int) pt.getXYCount(uval, y);
				}
			}
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					int denom = 0;
					for (int dval = 0; dval < m_ParamsPerAtt[u]; dval++) {
						denom += tempArray[dval][y];
					}
					double prob = Math.log(Math.max(1/m_ParamsPerAtt[u],1e-75));
					pt.setXYCount(uval, y, prob);
				}
			}

			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null) 					
					convertCountToUniformProbs(u, lparents, next);

				// Flag end of nodes
				att = -1;				
			}			
		}

		return;
	}


	/* 
	 * -----------------------------------------------------------------------------------------
	 *  Set counts/probs to Mutual information with the class of that particular attribute
	 * -----------------------------------------------------------------------------------------
	 */

	public void countsToMI() {
		for (int u = 0; u < n; u++) {
			setCountToMI(order[u], parents[u], wdBayesNode_[u]);
		}
	}

	public void setCountToMI(int u, int[] lparents, wdBayesNode pt) {

		int att = pt.att;

		if (att == -1) {
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					double prob = 1/mi[u];
					pt.setXYCount(uval, y, prob);
				}
			}			
			return;
		}

		while (att != -1) {

			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null) 					
					setCountToMI(u, lparents, next);

				// Flag end of nodes
				att = -1;				
			}			
		}

		return;
	}
	
	/* 
	 * -----------------------------------------------------------------------------------------
	 *  Set counts/probs to the log of Mutual information with the class of that particular attribute
	 * -----------------------------------------------------------------------------------------
	 */

	public void countsToLogMI() {
		for (int u = 0; u < n; u++) {
			setCountToLogMI(order[u], parents[u], wdBayesNode_[u]);
		}
	}

	public void setCountToLogMI(int u, int[] lparents, wdBayesNode pt) {

		int att = pt.att;

		if (att == -1) {
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					double prob = Math.log(1/mi[u]);
					pt.setXYCount(uval, y, prob);
				}
			}			
			return;
		}

		while (att != -1) {

			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null) 					
					setCountToLogMI(u, lparents, next);

				// Flag end of nodes
				att = -1;				
			}			
		}

		return;
	}
	
	/* 
	 * -----------------------------------------------------------------------------------------
	 *  Set counts/probs to the Full Mutual information with the class of that particular attribute
	 * -----------------------------------------------------------------------------------------
	 */

	public void countsToFullMI() {
		for (int u = 0; u < n; u++) {
			setCountToFullMI(order[u], parents[u], wdBayesNode_[u]);
		}
	}

	public void setCountToFullMI(int u, int[] lparents, wdBayesNode pt) {

		int att = pt.att;

		if (att == -1) {
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					double prob = 1/fmi[u][uval];
					pt.setXYCount(uval, y, prob);
				}
			}			
			return;
		}

		while (att != -1) {

			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null) 					
					setCountToFullMI(u, lparents, next);

				// Flag end of nodes
				att = -1;				
			}			
		}

		return;
	}
	
	/* 
	 * -----------------------------------------------------------------------------------------
	 *  Set counts/probs to the log of Full Mutual information with the class of that particular attribute
	 * -----------------------------------------------------------------------------------------
	 */

	public void countsToLogFullMI() {
		for (int u = 0; u < n; u++) {
			setCountToLogFullMI(order[u], parents[u], wdBayesNode_[u]);
		}
	}

	public void setCountToLogFullMI(int u, int[] lparents, wdBayesNode pt) {

		int att = pt.att;

		if (att == -1) {
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					double prob = Math.log(1/fmi[u][uval]);
					pt.setXYCount(uval, y, prob);
				}
			}			
			return;
		}

		while (att != -1) {

			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null) 					
					setCountToLogFullMI(u, lparents, next);

				// Flag end of nodes
				att = -1;				
			}			
		}

		return;
	}

	/* 
	 * -----------------------------------------------------------------------------------------
	 *  Compute Mutual Information
	 * -----------------------------------------------------------------------------------------
	 */

	public void computeMI() {
		for (int u = 0; u < n; u++) {
			mi[u] = computeMIForAtt(order[u], parents[u], wdBayesNode_[u]);
		}
	}

	private double computeMIForAtt(int u, int[] lparents, wdBayesNode pt) {

		double mia = 0;
		int att = pt.att;

		if (att == -1) {

			int[] tempArray = new int[m_ParamsPerAtt[u]];
			for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
				for (int y = 0; y < nc; y++) {
					tempArray[uval] += (int) pt.getXYCount(uval, y);
				}
			}

			for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
				for (int y = 0; y < nc; y++) {
					int avyCount = (int) pt.getXYCount(uval, y);
					if (avyCount > 0) {
						
						mia += (avyCount / (double)N) * Math.log(avyCount / (tempArray[uval]/(double)N * classCounts[y]) ) / Math.log(2);
					}
				}
			}			
			return mia;
		}

		while (att != -1) {

			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null) 					
					computeMIForAtt(u, lparents, next);

				// Flag end of nodes
				att = -1;				
			}			
		}

		return 0;
	}
	
	/* 
	 * -----------------------------------------------------------------------------------------
	 *  Compute Mutual Information
	 * -----------------------------------------------------------------------------------------
	 */

	public void computeFullMI() {
		for (int u = 0; u < n; u++) {
			fmi[u] = computeFullMIForAtt(order[u], parents[u], wdBayesNode_[u]);
		}
	}

	private double[] computeFullMIForAtt(int u, int[] lparents, wdBayesNode pt) {

		double[] mia = new double[m_ParamsPerAtt[u]];
		int att = pt.att;

		if (att == -1) {

			int[] tempArray = new int[m_ParamsPerAtt[u]];
			for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
				for (int y = 0; y < nc; y++) {
					tempArray[uval] += (int) pt.getXYCount(uval, y);
				}
			}

			for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
				
				int tempMargCount = 0;
				for (int uval2 = 0; uval2 < m_ParamsPerAtt[u]; uval2++) {
					if (uval != uval2) {
						for (int y = 0; y < nc; y++) {
							tempMargCount += (int) pt.getXYCount(uval2, y);
						}
					}
				}
				
				for (int y = 0; y < nc; y++) {
					
					int tempCount = 0;
					for (int uval2 = 0; uval2 < m_ParamsPerAtt[u]; uval2++) {
						if (uval != uval2) {
							tempCount += (int) pt.getXYCount(uval2, y);
							
						}
					}
					
					int avyCount = (int) pt.getXYCount(uval, y);
					if (avyCount > 0) {
						mia[uval] += (avyCount / (double)N) * Math.log(avyCount / (tempArray[uval]/(double)N * classCounts[y]) ) / Math.log(2);
					}
					if (tempCount > 0) {
						mia[uval] += (tempCount / (double)N) * Math.log(tempCount / (tempMargCount/(double)N * classCounts[y]) ) / Math.log(2);
					}
				}
			}			
			return mia;
		}

		while (att != -1) {

			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null) 					
					computeFullMIForAtt(u, lparents, next);

				// Flag end of nodes
				att = -1;				
			}			
		}

		return mia;
	}



	/* 
	 * -----------------------------------------------------------------------------------------
	 * xyCount are updated and 
	 * xyParameters are initialized to zero
	 * -----------------------------------------------------------------------------------------
	 */

	public void initialize(Instance instance) {
		classCounts[(int) instance.classValue()]++;

		for (int u = 0; u < n; u++) {
			updateTrieStructure(instance, u, order[u], parents[u]);
		}

		N++;
	}

	public void updateTrieStructure(Instance instance, int i, int u, int[] lparents) {
		int x_C = (int) instance.classValue();
		int x_u = (int) instance.value(u);		

		wdBayesNode_[i].setXYParameter(x_u, x_C, 0);	
		wdBayesNode_[i].incrementXYCount(x_u, x_C);

		if (lparents != null) {

			wdBayesNode currentdtNode_ = wdBayesNode_[i];

			for (int d = 0; d < lparents.length; d++) {
				int p = lparents[d];

				if (currentdtNode_.att == -1 || currentdtNode_.children == null) {
					currentdtNode_.children = new wdBayesNode[m_ParamsPerAtt[p]];
					currentdtNode_.att = p;	
					currentdtNode_.index = -1;
				}

				int x_up = (int) instance.value(p);
				currentdtNode_.att = p;
				currentdtNode_.index = -1;

				// the child has not yet been allocated, so allocate it
				if (currentdtNode_.children[x_up] == null) {
					currentdtNode_.children[x_up] = new wdBayesNode(scheme);
					currentdtNode_.children[x_up].init(nc, m_ParamsPerAtt[u]);
				} 

				currentdtNode_.children[x_up].setXYParameter(x_u, x_C, 0);
				currentdtNode_.children[x_up].incrementXYCount(x_u, x_C);

				currentdtNode_ = currentdtNode_.children[x_up];
			}

		}
	}

	/* 
	 * -----------------------------------------------------------------------------------------
	 * Allocate Parameters
	 * -----------------------------------------------------------------------------------------
	 */	

	public void allocate() {
		// count active nodes in Trie
		np = nc;
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			activeNumNodes[u] = countActiveNumNodes(u, order[u], parents[u], pt);
		}		
		System.out.println("Allocating dParameters of size: " + np);
		parameters = new double[np];				
	}

	public int countActiveNumNodes(int i, int u, int[] lparents, wdBayesNode pt) {
		int numNodes = 0;		
		int att = pt.att;

		if (att == -1) {
			pt.index = np;
			np += m_ParamsPerAtt[u] * nc;			
			return 1;			
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					numNodes += countActiveNumNodes(i, u, lparents, next);
				att = -1;
			}			
		}

		return numNodes;
	}

	/* 
	 * -----------------------------------------------------------------------------------------
	 * xyParameters to Parameters
	 * -----------------------------------------------------------------------------------------
	 */	

	public void reset() {		
		// convert a trie into an array
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			trieToArray(u, order[u], parents[u], pt);
		}		
	}

	private int trieToArray(int i, int u, int[] parents, wdBayesNode pt) {		
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					//System.out.println(index + (c * paramsPerAtt[u] + j));
					parameters[index + (c * m_ParamsPerAtt[u] + j)] = pt.getXYParameter(j, c);
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					trieToArray(i, u, parents, next);
				att = -1;
			}			
		}

		return 0;		
	}

	// ----------------------------------------------------------------------------------
	// Parameters to xyParameters
	// ----------------------------------------------------------------------------------

	public void copyParameters(double[] params) {
		for (int i = 0; i < params.length; i++) {
			parameters[i] = params[i];
		}		

		// convert an array into a trie
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			arrayToTrie(u, order[u], parents[u], pt);			
		}		
	}

	private int arrayToTrie(int i, int u, int[] parents, wdBayesNode pt) {
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					double val = parameters[index + (c * m_ParamsPerAtt[u] + j)];
					pt.setXYParameter(j, c, val);
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					arrayToTrie(i, u, parents, next);
				att = -1;
			}			
		}

		return 0;	
	}

	public void multiplyParametersToProbs(double[] params) {	
		for (int c = 0; c < nc; c++) {
			classCounts[c] = classCounts[c] * parameters[c];
		}

		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			multiplyParametersToProbs(u, order[u], parents[u], pt);			
		}		
	}

	private int multiplyParametersToProbs(int i, int u, int[] parents, wdBayesNode pt) {
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					double val1 = parameters[index + (c * m_ParamsPerAtt[u] + j)];
					double val2 = pt.getXYCount(j, c);
					pt.setXYCount(j, c, val1 *val2);
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					multiplyParametersToProbs(i, u, parents, next);
				att = -1;
			}			
		}

		return 0;	
	}


	/* 
	 * -----------------------------------------------------------------------------------------
	 * Initialize Parameters
	 * -----------------------------------------------------------------------------------------
	 */	

	public void putProbsInBuffer() {
		for (int c = 0; c < nc; c++) {
			classBuffer[c] = classCounts[c];
		}
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			putProbsInBuffer(u, order[u], parents[u], pt);
		}
	}

	private int putProbsInBuffer(int i, int u, int[] parents, wdBayesNode pt) {		
		int att = pt.att;

		if (att == -1) {
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					double val = pt.getXYCount(j, c);	
					pt.setXYBuffer(j, c, val);
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					putProbsInBuffer(i, u, parents, next);
				att = -1;
			}			
		}

		return 0;		
	}

	public void putBufferInParameters() {
		for (int c = 0; c < nc; c++) {
			parameters[c] = classBuffer[c];
		}
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			putBufferInParameters(u, order[u], parents[u], pt);
		}
	}

	private int putBufferInParameters(int i, int u, int[] parents, wdBayesNode pt) {		
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					double val = pt.getXYBuffer(j, c);	
					pt.setXYParameter(j, c, val);
					parameters[index + (c * m_ParamsPerAtt[u] + j)] = pt.getXYParameter(j, c);
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					putBufferInParameters(i, u, parents, next);
				att = -1;
			}			
		}

		return 0;		
	}

	public void divideProbsByParameters() {
		for (int c = 0; c < nc; c++) {
			classCounts[c] = classCounts[c] / parameters[c];
		}
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			divideProbsByParameters(u, order[u], parents[u], pt);
		}
	}

	private int divideProbsByParameters(int i, int u, int[] parents, wdBayesNode pt) {		
		int att = pt.att;

		if (att == -1) {
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					double val = pt.getXYCount(j, c) / pt.getXYParameter(j, c);	
					pt.setXYCount(j, c, val);
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					divideProbsByParameters(i, u, parents, next);
				att = -1;
			}			
		}

		return 0;		
	}

	public void divideParametersByProbs() {
		for (int c = 0; c < nc; c++) {
			parameters[c] = parameters[c] / classCounts[c];
		}
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			divideParametersByProbs(u, order[u], parents[u], pt);
		}
	}

	private int divideParametersByProbs(int i, int u, int[] parents, wdBayesNode pt) {		
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					double val = pt.getXYParameter(j, c) / pt.getXYCount(j, c);	
					pt.setXYParameter(j, c, val);
					parameters[index + (c * m_ParamsPerAtt[u] + j)] = val;
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					divideParametersByProbs(i, u, parents, next);
				att = -1;
			}			
		}

		return 0;		
	}

	public void initializeParameters(int m_WeightingInitialization) {
		//if (m_WeightingInitialization == -1) {
		//	initializeParametersWithProbs();
		//} else {
		initializeParametersWithVal(m_WeightingInitialization);
		//}

	}

	public void initializeParametersWithProbs() {
		for (int c = 0; c < nc; c++) {
			parameters[c] = classCounts[c];
		}
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			initializeParametersWithProbs(u, order[u], parents[u], pt);
		}
	}

	private int initializeParametersWithProbs(int i, int u, int[] parents, wdBayesNode pt) {		
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					double val = pt.getXYCount(j, c);	
					pt.setXYParameter(j, c, val);
					parameters[index + (c * m_ParamsPerAtt[u] + j)] = pt.getXYParameter(j, c);
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					initializeParametersWithProbs(i, u, parents, next);
				att = -1;
			}			
		}

		return 0;		
	}

	private void initializeParametersWithVal(double initVal) {
		for (int c = 0; c < nc; c++) {
			parameters[c] = initVal;
		}
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			initializeParametersWithVal(u, order[u], parents[u], pt, initVal);
		}
	}

	private int initializeParametersWithVal(int i, int u, int[] parents, wdBayesNode pt, double initVal) {		
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					pt.setXYParameter(j, c, initVal);	
					parameters[index + (c * m_ParamsPerAtt[u] + j)] = initVal;
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					initializeParametersWithProbs(i, u, parents, next);
				att = -1;
			}			
		}

		return 0;		
	}

	public void initializeProbsWithParameters() {
		for (int c = 0; c < nc; c++) {
			classCounts[c] = parameters[c];
		}
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			initializeProbsWithParameters(u, order[u], parents[u], pt);
		}
	}

	private int initializeProbsWithParameters(int i, int u, int[] parents, wdBayesNode pt) {		
		int att = pt.att;

		if (att == -1) {
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					double val = pt.getXYParameter(j, c);
					pt.setXYCount(j, c, val);
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					initializeParametersWithProbs(i, u, parents, next);
				att = -1;
			}			
		}

		return 0;		
	}

	public void scaleProbsWithFactor(double factor) {
		for (int c = 0; c < nc; c++) {
			classCounts[c] *= factor;
		}
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			scaleProbsWithFactor(u, order[u], parents[u], pt, factor);
		}
	}

	private int scaleProbsWithFactor(int i, int u, int[] parents, wdBayesNode pt, double factor) {		
		int att = pt.att;

		if (att == -1) {
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					double val = pt.getXYCount(j, c) * factor;
					pt.setXYCount(j, c, val);
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					scaleProbsWithFactor(i, u, parents, next, factor);
				att = -1;
			}			
		}

		return 0;		
	}

	// ----------------------------------------------------------------------------------
	// Access Functions
	// ----------------------------------------------------------------------------------

	public double getClassParameter(int c) {		
		return parameters[c];
	}

	public double getXYParameter(Instance instance, int c, int i, int u, int[] m_Parents) {	

		wdBayesNode pt = wdBayesNode_[i];
		int att = pt.att;

		// find the appropriate leaf
		while (att != -1) {
			int v = (int) instance.value(att);
			wdBayesNode next = pt.children[v];
			if (next == null) 
				break;
			pt = next;
			att = pt.att;
		}

		return pt.getXYParameter((int)instance.value(u), c);	
	}

	public int getClassIndex(int k) {		
		return k;
	}

	public int getXYParameterIndex(Instance instance, int c, int i, int u, int[] m_Parents) {

		wdBayesNode pt = wdBayesNode_[i];
		int att = pt.att;

		// find the appropriate leaf
		while (att != -1) {
			int v = (int) instance.value(att);
			wdBayesNode next = pt.children[v];
			if (next == null) 
				break;
			pt = next;
			att = pt.att;
		}

		return pt.getXYIndex((int)instance.value(u), c);	
	}

	public wdBayesNode getBayesNode(Instance instance, int i, int u, int[] m_Parents) {	

		wdBayesNode pt = wdBayesNode_[i];
		int att = pt.att;

		// find the appropriate leaf
		while (att != -1) {
			int v = (int) instance.value(att);
			wdBayesNode next = pt.children[v];
			if (next == null) 
				break;
			pt = next;
			att = pt.att;
		}

		return pt;		
	}

	public wdBayesNode getBayesNode(Instance instance, int i) {	

		wdBayesNode pt = wdBayesNode_[i];
		int att = pt.att;

		// find the appropriate leaf
		while (att != -1) {
			int v = (int) instance.value(att);
			wdBayesNode next = pt.children[v];
			if (next == null) 
				break;
			pt = next;
			att = pt.att;
		}
		return pt;		
	}

	// ----------------------------------------------------------------------------------
	// Others
	// ----------------------------------------------------------------------------------

	public void setParametersToOne() {
		Arrays.fill(parameters, 1.0);
	}

	public double[] getParameters() {
		return parameters;
	}

	public int getNp() {
		return np;
	}

	public int getNAttributes() {
		return n;
	}

	public double getNLL_MAP(Instances instances) {

		double nll = 0;
		int N = instances.numInstances();
		double mLogNC = -Math.log(nc); 
		double[] myProbs = new double[nc];

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			// unboxed logDistributionForInstance_d(instance,nodes);
			for (int c = 0; c < nc; c++) {
				myProbs[c] = classCounts[c];
				//myProbs[c] = xyDist.pp(c);
			}
			for (int u = 0; u < n; u++) {
				wdBayesNode bNode = getBayesNode(instance, u);
				for (int c = 0; c < nc; c++) {
					myProbs[c] += bNode.getXYCount((int) instance.value(order[u]), c);
				}
			}
			SUtils.normalizeInLogDomain(myProbs);
			nll += (mLogNC - myProbs[x_C]);
			//nll += (- myProbs[x_C]);
		}

		return nll;
	}

	public double getNLL_dCCBN(Instances instances) {
		double nll = 0;
		int N = instances.numInstances();
		double mLogNC = -Math.log(nc); 
		double[] myProbs = new double[nc];

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			// unboxed logDistributionForInstance_d(instance,nodes);
			for (int c = 0; c < nc; c++) {
				myProbs[c] = getClassParameter(c);
			}
			for (int u = 0; u < n; u++) {
				//wdBayesNode bNode = myNodes[u];
				wdBayesNode bNode = getBayesNode(instance, u);
				for (int c = 0; c < nc; c++) {
					myProbs[c] += bNode.getXYParameter((int) instance.value(order[u]), c);
				}
			}
			SUtils.normalizeInLogDomain(myProbs);
			nll += (mLogNC - myProbs[x_C]);
			//nll += (- myProbs[x_C]);
		}

		return nll;
	}

	public double getNLL_eCCBN(Instances instances) {

		double nll = 0;
		int N = instances.numInstances();
		double mLogNC = -Math.log(nc); 
		double[] myProbs = new double[nc];

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			for (int c = 0; c < nc; c++) {
				myProbs[c] = getClassParameter(c);
			}
			for (int u = 0; u < n; u++) {
				wdBayesNode bNode = getBayesNode(instance, u);
				for (int c = 0; c < nc; c++) {
					myProbs[c] += bNode.getXYParameter((int) instance.value(order[u]), c);
				}
			}
			SUtils.normalizeInLogDomain(myProbs);
			nll += (mLogNC - myProbs[x_C]);
		}

		return nll;
	}

	public double getNLL_wCCBN(Instances instances) {

		double nll = 0;
		int N = instances.numInstances();
		double mLogNC = -Math.log(nc); 
		double[] myProbs = new double[nc];

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			// unboxed logDistributionForInstance_d(instance,nodes);
			for (int c = 0; c < nc; c++) {
				//myProbs[c] = xyDist.pp(c) * getClassParameter(c);
				myProbs[c] = classCounts[c] * getClassParameter(c);
			}
			for (int u = 0; u < n; u++) {
				wdBayesNode bNode = getBayesNode(instance, u);
				for (int c = 0; c < nc; c++) {
					myProbs[c] += bNode.getXYParameter((int) instance.value(order[u]), c) * bNode.getXYCount((int) instance.value(order[u]), c);
				}
			}
			SUtils.normalizeInLogDomain(myProbs);
			nll += (mLogNC - myProbs[x_C]);
			//nll += (- myProbs[x_C]);
		}

		return nll;
	}

	public double[] getClassCounts() {
		return classCounts;
	}

	public double[] getMutualInfomation() {
		return mi;
	}

	public double[][] getFullMutualInfomation() {
		return fmi;
	}

	public void printProbabilities() {
		//		System.out.println();
		//		
		//		for (int c = 0; c < nc; c++) {
		//			//System.out.print(xxyDist_.xyDist_.getClassCount(c) + ", ");
		//		}
		//		
		//		for (int u = 0; u < n; u++) {
		//			for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
		//				for (int c = 0; c < nc; c++) {
		//					System.out.print("P(x_" + u + "=" + uval + " | y = " + c + ") = " + xxyDist_.xyDist_.getCount(u, uval, c) + ",   " );
		//				}
		//				System.out.println();
		//			}
		//		}

	}

}