import java.util.Random;
import java.util.Arrays;
import java.util.Collections;

public class NNLearner {
	
	State s = new State();
	State p = new State();
	State goal = new State();
	NeuralNet nn = new NeuralNet();
	int x, y, z;
	int action; // N(0), E(1), S(2), W(3)
	double [][] reward;
	
	
	//Default Constructor
	NNLearner(double[][] reward, State goal, Random r)
	{
		x = 20;
		y = 10;
		z = 4;
		this.reward = reward;
		this.goal = goal;
		nn.layers.add(new LayerTanh(2, 20));
		nn.layers.add(new LayerTanh(20, 4));
		nn.init(r);
	}
	
	//Specify Size of Q-table
	NNLearner(int x, int y, int possibleMoves, double[][] reward, State goal, Random r)
	{
		this.x = x;
		this.y = y;
		z = possibleMoves;
		this.reward = reward;
		this.goal = goal;
		nn.layers.add(new LayerTanh(2, 20));
		nn.layers.add(new LayerTanh(20, 4));
		nn.init(r);
	}
	
	void do_action(int a, Random r)
	{
		// N(0), E(1), S(2), W(3)
		p.x = s.x;
		p.y = s.y;
		//int b = a;
		if(r.nextDouble() < 0.01)
			do
				a = r.nextInt(4);
			while(!actionIsValid(a));
		/*if(b != a)
			System.out.println("Random");
		else
			System.out.println("NOT");*/
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
	
	//Find best action for every state and print it out
	void print()
	{
		int [][] bestAction = new int[20][10];
		for(int x = 0; x < 20; x++)
		{
			for(int y = 0; y < 10; y++)
			{
				bestAction[x][y] = 0;
				double[] best = new double[2];
				best[0] = (double)x / 19;
				best[1] = (double)y / 9;
				double[] qValues = nn.forwardProp(best);
				System.out.println(qValues[0] + qValues[1] + qValues[2] + qValues[3]);
				for(int i = bestAction[x][y]; i < 4; i++)
					if(qValues[i] > qValues[bestAction[x][y]])
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
		
		//get qvalues for current state
		double[] curState = new double[2];
		curState[0] = (double)s.x / 19;
		curState[1] = (double)s.y / 9;
		double[] prediction = nn.forwardProp(curState);
		
		//Check if all options are equal
		boolean different = false;
		for(int i = 0; i < z-1; i++)
			if(prediction[i] != prediction[i+1])
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
				if(prediction[i] > prediction[action])
					if(actionIsValid(i))
						action = i;
		}
		double[] copy = Vec.copy(prediction);
		double[] prevState = Vec.copy(curState);
		do_action(action, r);
		//Learn from the action (s is current state, p is previous state)

		int b = 0; //Best Next Action from s
		double gamma = 0.97; //Reduction from next state
		//get qvalues for current state
		curState[0] = (double)s.x / 19;
		curState[1] = (double)s.y / 9;
		double[] predictionb = nn.forwardProp(curState);
		while(!actionIsValid(b))
			b++;
		for(int i = b; i < z; i++)
			if(predictionb[i] > predictionb[b])
				if(actionIsValid(i))
					b = i;
		//System.out.println("Reward: " + reward[s.x][s.y]);
		//System.out.println("Gamma * Prediction: " + gamma * predictionb[b]);
		//System.out.println(b);
		if(s.equals(goal))
			predictionb[b] = 0;
		copy[action] = (reward[s.x][s.y] + gamma * predictionb[b]);
		//System.out.println(curState[0] + " " + curState[1] + " " + copy[0] + " " + copy[1] + " " + copy[2] + " " + copy[3]);
		nn.trainIncremental(prevState, copy, 0.03);
		//System.out.println("HERE");
		if(s.equals(goal))
			s = new State();
		
	}

}
