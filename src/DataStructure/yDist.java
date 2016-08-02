package DataStructure;

import java.util.Vector;

import Utils.SUtils;
import weka.core.Instance;

public class yDist {

	private Vector<Integer> counts = new Vector<Integer>();
	private int total = 0;

	public yDist(int nc) {
		counts.setSize(nc);
	}

	public double p(int y) {
		return SUtils.MEsti(counts.get(y), total, counts.size());
	}

	public double rawP(int y) {
		return counts.get(y) / total;
	}

	public int count(int y) {
		return counts.get(y);
	}

	void update(Instance inst) {
		int x_C = (int)inst.classValue();
		int val = counts.get(x_C);
		counts.add(x_C, val+1);
		total++;
	}

	public int getNoClasses() { 
		return counts.size(); 
	}

}
