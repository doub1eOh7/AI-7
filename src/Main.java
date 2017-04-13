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
			reward[10][y] = -0.1;
		reward[10][4] = 0;
		reward[10][5] = 0;
		QLearner q = new QLearner(reward, goal);
		for(int i = 0; i < 10000000; i++)
			q.learn(r);
		q.print();
		//q.printQ();
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
