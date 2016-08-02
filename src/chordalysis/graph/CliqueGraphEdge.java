package chordalysis.graph;

import java.util.BitSet;

public class CliqueGraphEdge {
	BitSet c1,c2,separator;
	public CliqueGraphEdge(BitSet c1,BitSet c2){
		this.c1 = c1;
		this.c2 = c2;
		this.separator = (BitSet) c1.clone();
		this.separator.and(c2);
	}
	
	@Override
	public boolean equals(Object o) {
		if (o == this)
            return true;
		if(o instanceof CliqueGraphEdge){
			CliqueGraphEdge e = (CliqueGraphEdge) o;
			
			return (c1.size()==e.c1.size() && c2.size()==e.c2.size() && e.c1.equals(c1) && e.c2.equals(c2))||
					(c1.size()==e.c2.size() && c2.size()==e.c1.size() && e.c1.equals(c2) && e.c2.equals(c1));
		}else{
			return false;
		}
	}
	
	@Override
	public String toString(){
		return c1.toString()+" inter "+c2.toString()+" = "+separator.toString();
	}
	
	public BitSet getClique1(){
		return (BitSet) c1.clone();
	}
	
	public BitSet getClique2(){
		return (BitSet) c2.clone();
	}
	
	public BitSet getSeparator(){
		return (BitSet) separator.clone();
	}

}
