import java.util.Random;
import java.util.Arrays;
import java.util.Collections;

public class QLearner {
	
	State s = new State();
	State p = new State();
	State goal = new State();
	double [][][] qTable; //X dimension, Y dimension, and number of possible moves
	int x, y, z;
	int action; // N(0), E(1), S(2), W(3)
	double [][] reward;
	
	
	//Default Constructor
	QLearner(double[][] reward, State goal)
	{
		qTable = new double[20][10][4];
		x = 20;
		y = 10;
		z = 4;
		this.reward = reward;
		this.goal = goal;
	}
	
	//Specify Size of Q-table
	QLearner(int x, int y, int possibleMoves, double[][] reward, State goal)
	{
		qTable = new double[x][y][possibleMoves];
		this.x = x;
		this.y = y;
		z = possibleMoves;
		this.reward = reward;
		this.goal = goal;
	}
	
	void do_action(int a)
	{
		// N(0), E(1), S(2), W(3)
		p.x = s.x;
		p.y = s.y;
		switch(a){
		case 0:
			s.y--;
			break;
		case 1:
			s.x++;
			break;
		case 2:
			s.y++;
			break;
		case 3:
			s.x--;
			break;
		}
			
	}
	boolean actionIsValid(int a)
	{
		if(a == 0 && s.y == 0)
			return false;
		else if(a == 1 && s.x == x-1)
			return false;
		else if(a == 2 && s.y == y-1)
			return false;
		else if(a == 3 && s.x == 0)
			return false;
		else
			return true;
	}
	
	void printQ()
	{
		for(int y = 0; y < 10; y++)
		{
			for(int x = 0; x < 20; x++)
			{
				double bestValue = qTable[x][y][0];
				for(int z = 1; z < 4; z++)
					if(qTable[x][y][z] > bestValue)
						bestValue = qTable[x][y][z];
				System.out.print(bestValue + " ");
			}
			System.out.println("\n");
		}
	}
	
	void printAllQ()
	{
		for(int z = 0; z < 4; z++)
		{
			for(int y = 0; y < 10; y++)
			{
				for(int x = 0; x < 20; x++)
				{
					System.out.print(qTable[x][y][z] + " ");
				}
				System.out.println("\n");
			}
			System.out.println("\n\n\n");
		}
	}
	
	//Find best action for every state and print it out
	void print()
	{
		int [][] bestAction = new int[20][10];
		for(int x = 0; x < 20; x++)
		{
			for(int y = 0; y < 10; y++)
			{
				bestAction[x][y] = 0;
				for(int i = bestAction[x][y]; i < 4; i++)
					if(qTable[x][y][i] > qTable[x][y][bestAction[x][y]])
						bestAction[x][y] = i;
			}
		}
		
		for(int y = 0; y < 10; y++)
		{
			for(int x = 0; x < 20; x++)
			{
				if((x == 10 && y > 5) || (x == 10 && y < 4))
					System.out.print("# ");
				else if(x == 0 && y == 9)
					System.out.print("S ");
				else if(x == 19 && y == 0)
					System.out.print("G ");
				else if(bestAction[x][y] == 0)
					System.out.print("^ ");
				else if(bestAction[x][y] == 1)
					System.out.print("> ");
				else if(bestAction[x][y] == 2)
					System.out.print("V ");
				else if(bestAction[x][y] == 3)
					System.out.print("< ");
			}
			System.out.println("\n");
		}
		
	}
	
	void learn(Random r)
	{
		double e = 0.05;
		
		//Check if all options are equal
		boolean different = false;
		for(int i = 0; i < z-1; i++)
			if(qTable[s.x][s.y][i] != qTable[s.x][s.y][i+1])
				different = true;
		
		if(r.nextDouble() < e || !different)
			//Explore (choose a random valid action)
			do
				action = r.nextInt(z);
			while (!actionIsValid(action));
		else
		{
			//Exploit (choose the best valid action)
			action = 0;
			while(!actionIsValid(action))
				action++;
			for(int i = action; i < z; i++)
				if(qTable[s.x][s.y][i] > qTable[s.x][s.y][action])
					if(actionIsValid(i))
						action = i;
		}
		
		do_action(action);
		//Learn from the action (s is current state, p is previous state)
		int a = action; //Action taken from p
		int b = 0; //Best Next Action from s
		double alphaK = 0.1;
		double gamma = 0.97; //Reduction from next state
		while(!actionIsValid(b))
			b++;
		for(int i = b; i < z; i++)
			if(qTable[s.x][s.y][i] > qTable[s.x][s.y][b])
				if(actionIsValid(i))
					b = i;
		
		qTable[p.x][p.y][a] = (1 - alphaK) * qTable[p.x][p.y][a] + alphaK * (reward[s.x][s.y] + gamma * qTable[s.x][s.y][b]);
		
		if(s.equals(goal))
			s = new State();
		
	}

}
