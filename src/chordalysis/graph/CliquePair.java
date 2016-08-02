package chordalysis.graph;

import java.util.TreeSet;

public class CliquePair {
	public TreeSet<Integer> c1;
	public TreeSet<Integer> c2;
	public CliquePair(TreeSet<Integer>c1,TreeSet<Integer>c2){
		this.c1 = c1;
		this.c2 = c2;
	}
	
	@Override
	public String toString(){
		return "("+c1.toString()+" , "+c2.toString()+")";
	}
}
