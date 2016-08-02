package DataStructure;

/*
* wdBayesNode - Node of Trie, storing discriminative parameters
*  
* dParameterTree constitutes of wdBayesNodes
*
*/
public class wdBayesNode {

	public double[] xyParameter;  	// Parameter indexed by x val the y val
	public double[] xyCount;		// Count for x val and the y val
	public double[] xyProbability;		// Count for x val and the y val

	public double[] xyBuffer;		// Count for x val and the y val
	
	wdBayesNode[] children;	

	public int att;          // the Attribute whose values select the next child
	public int index;
	private int paramsPerAttVal;

	public int scheme;

	// default constructor - init must be called after construction
	public wdBayesNode(int s1) {
		scheme = s1;
	}     

	// Initialize a new uninitialized node
	public void init(int nc, int paramsPerAttVal) {
		this.paramsPerAttVal = paramsPerAttVal;
		index = -1;
		att = -1;

		if (scheme == 1) {
			xyCount = new double[nc * paramsPerAttVal];
		}  else if (scheme == 2) {
			xyParameter = new double[nc * paramsPerAttVal];
			xyCount = new double[nc * paramsPerAttVal];
			//xyProbability = new double[nc * paramsPerAttVal];
			xyBuffer = new double[nc * paramsPerAttVal];
		}			

		children = null;
	}  

	// Reset a node to be empty
	public void clear() { 
		att = -1;
		xyParameter = null;
		children = null;
	}      

	public void setXYParameter(int v, int y, double val) {
		xyParameter[y * paramsPerAttVal + v] = val;
	}

	public double getXYParameter(int v, int y) {
		return xyParameter[y * paramsPerAttVal + v];		
	}

	public void setXYCount(int v, int y, double val) {
		xyCount[y * paramsPerAttVal + v] = val;
	}

	public double getXYCount(int v, int y) {
		return xyCount[y * paramsPerAttVal + v];		
	}
	
	public void setXYProbability(int v, int y, double val) {
		xyProbability[y * paramsPerAttVal + v] = val;
	}

	public double getXYProbability(int v, int y) {
		return xyProbability[y * paramsPerAttVal + v];		
	}

	public int getXYIndex(int v, int y) {
		return index + (y * paramsPerAttVal + v);
	}
	
	public void setXYBuffer(int v, int y, double val) {
		xyBuffer[y * paramsPerAttVal + v] = val;
	}

	public double getXYBuffer(int v, int y) {
		return xyBuffer[y * paramsPerAttVal + v];		
	}

	public void incrementXYCount(int v, int y) {
		xyCount[y * paramsPerAttVal + v]++;
	}

}