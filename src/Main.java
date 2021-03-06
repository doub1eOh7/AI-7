import java.util.Random;
public class Main {
	
	public static void main(String[] args)
	{
		Random r = new Random(0);
		State goal = new State(19, 0);
		double [][] reward = new double[20][10];
		for(int x = 0; x < 20; x++)
			for(int y = 0; y < 10; y++)
				reward[x][y] = 0;
		reward[goal.x][goal.y] = 1;
		for(int y = 0; y < 10; y++)
			reward[10][y] = -0.01;
		reward[10][4] = 0;
		reward[10][5] = 0;
		
		//Do neural network
		if(args.length > 0)
		{
			if(args[0].equals("nn"))
			{
				System.out.println("Using Neural Network");
				NNLearner nn = new NNLearner(reward, goal, r);
				for(int i = 0; i < 10000000; i++)
				{
					if(i % 100000 == 0)
						System.out.println(i/100000 + "% Complete");
					nn.learn(r);
				}
				nn.print();

			}
		}
		//Do q learner
		else
		{
			QLearner q = new QLearner(reward, goal);
			for(int i = 0; i < 100000000; i++)
				q.learn(r);
			q.print();
			//q.printQ();
		}
		//q.printAllQ();
		/*
		for(int y = 0; y < 10; y++)
		{
			for(int x = 0; x < 20; x++)
			{
				System.out.print(reward[x][y] + " ");
			}
			System.out.println();
		}
		*/
	}

}
