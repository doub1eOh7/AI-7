
public class State {
	
	int x, y;
	
	State()
	{
		x = 0;
		y = 9;
	}
	
	State(int x, int y)
	{
		this.x = x;
		this.y = y;
	}
	
	void setLocation(int x, int y)
	{
		this.x = x;
		this.y = y;
	}
	
	@Override
	public boolean equals(Object obj)
	{
		if(obj == null)
			return false;
		else if(!State.class.isAssignableFrom(obj.getClass()))
			return false;
		final State other = (State) obj;
		if(this.x != other.x || this.y != other.y)
			return false;
		else
			return true;
	}

}
