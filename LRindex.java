

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class LRindex {
	/* HashMap for distinct words */
	public static LinkedHashMap<String, Integer> vocabDir = new LinkedHashMap<String, Integer>();
	
	/* HashMap words in ham/spam */
	public static LinkedHashMap<String, Integer> hamVocabDir = new LinkedHashMap<String, Integer>();
	public static LinkedHashMap<String, Integer> spamVocabDir = new LinkedHashMap<String, Integer>();

	/* HashMap for weights */
	public static LinkedHashMap<String, Double> weighted_word = new LinkedHashMap<String, Double>();
	public static LinkedHashMap<String, Double> delta_weighted_word = new LinkedHashMap<String, Double>();
	
	
	/* HashMap for test dir */
	public static LinkedHashMap<String, Integer> testVocabDir = new LinkedHashMap<String, Integer>();
	public static LinkedHashMap<String, Integer> testHamVocabDir = new LinkedHashMap<String, Integer>();
	public static LinkedHashMap<String, Integer> testSpamVocabDir = new LinkedHashMap<String, Integer>();

	/* HashMap for test words */
	public static LinkedHashMap<Integer, LinkedHashMap<String, Integer>> testHamVocabMap = new LinkedHashMap<Integer, LinkedHashMap<String, Integer>>();
	public static LinkedHashMap<Integer, LinkedHashMap<String, Integer>> testSpamVocabMap = new LinkedHashMap<Integer, LinkedHashMap<String, Integer>>();
	
	/* HashMap for training words */
	public static LinkedHashMap<Integer, LinkedHashMap<String, Integer>> vocabMap = new LinkedHashMap<Integer, LinkedHashMap<String, Integer>>();
	public static LinkedHashMap<Integer, LinkedHashMap<String, Integer>> hamVocabMap = new LinkedHashMap<Integer, LinkedHashMap<String, Integer>>();
	public static LinkedHashMap<Integer, LinkedHashMap<String, Integer>> spamVocabMap = new LinkedHashMap<Integer, LinkedHashMap<String, Integer>>();
	
	/* Files set */
	public static ArrayList<File> allDocs = new ArrayList<File>();
	public static ArrayList<File> hamDocs = new ArrayList<File>();
	public static ArrayList<File> spamDocs = new ArrayList<File>();
	
	/* Doc list for test data */
	public static ArrayList<File> test_allDocs = new ArrayList<File>();
	public static ArrayList<File> test_hamDocs = new ArrayList<File>();
	public static ArrayList<File> test_spamDocs = new ArrayList<File>();

	public static double learning_rate=0.01;
	public static double lambda=0.001;
	
	public static double w0=0.1;
	public static int doc_count=0;
	public static int num_of_iterations=10;
	
	public static String HAM ="ham";
	public static String SPAM ="spam";
	public static String REGEX = "[\\w']+";
	
	/* Mcap lists */
	public static double Pr[];
	public static double w[];
	public static double dw[];
	
	
	/**
	 * @param args
	 * Argument 0 : path to folder containing train
	 * Argument 1 : path to folder containing test
	 * Argument 2: num_iterations
	 * Argument 3: learning_rate_eta
	 * Argument 4: lambda
	 */
	public static void main(String args[]) {
		try{
//			String mainDir="/Users/poorvitusam/Documents/workspace/naive_bayes/res/train/";
//			String testDir="/Users/poorvitusam/Documents/workspace/naive_bayes/res/test/";
			String mainDir=args[0];
			String testDir=args[1];
			
			num_of_iterations=Integer.parseInt(args[2]);
			learning_rate=Double.parseDouble(args[3]);
			lambda=Double.parseDouble(args[4]);
			
			
			getFiles(mainDir,allDocs, spamDocs, hamDocs);/* Read training */
			readTrainingData();
			MCAP(); /* Training using MCAP algorithm */
			doc_count=0;
			checkWithTest(testDir);/* Testing begins */
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Algorithm of MCAP::
	 * Initialise Pr containing the sigmoid value by a random number
	 * Initialise weights by a random number and delta weights by zero
	 * Calculate sigmoid for each doc
	 * Go over all the	weights for all training examples: 
	 * calculate delta weights=delta weights+frequency of training example * (class_of_file - sigmoid value of the file)
	 * Update weight: weight=weight+learning_rate*(delta_weight-lambda*weight*weight) Note:weight*weight for L2 regularization
	 */
	private static void MCAP() {
		int m=vocabMap.size();
		int n=vocabDir.size();
		
		for(int itr=0;itr<num_of_iterations;itr++) {
			Pr=new double[m];
			for(int i=0;i<m;i++) {
				double r =  (Math.random() * (1 -(-1))) + (-1);
				Pr[i]=r;
			}

			for (Entry<String, Integer> entry : vocabDir.entrySet()) {
				double r =  (Math.random() * (1 -(-1))) + (-1);
				weighted_word.put(entry.getKey(), r);
				delta_weighted_word.put(entry.getKey(), 0.0);
			}

			for(int j=0;j<m;j++) {
				Pr[j]=sigmoidOfDoc(j);
			}
			
			for (Entry<String, Integer> entry : vocabDir.entrySet()) {
				double delta_weight=0;
				for (Entry<Integer, LinkedHashMap<String, Integer>> file : vocabMap.entrySet()) {
					delta_weight=delta_weighted_word.get(entry.getKey());
					double freq=0;
					if(file.getValue().containsKey(entry.getKey())) {
						freq= (double)file.getValue().get(entry.getKey());
					}
					
					double class_of_file;
					if(hamVocabMap.containsKey(file.getKey())) {
						class_of_file=0.0;
					} else {
						class_of_file=1.0;
					}
					delta_weight=delta_weight+freq*(class_of_file-Pr[file.getKey()]);
					delta_weighted_word.put(entry.getKey(),delta_weight);
				}
				double weight=weighted_word.get(entry.getKey());
				weight=weight+learning_rate*(delta_weight-lambda*weight*weight);
			}
		}
	}
	
	private static double sigmoidOfDoc(int file_no) {
		if(hamVocabMap.containsKey(file_no)) {
			double weighted_sum=w0;
			for (Entry<String, Integer> entry : hamVocabMap.get(file_no).entrySet()) {
				weighted_sum += weighted_word.get(entry.getKey())*entry.getValue();
			}
			return sigmod(weighted_sum, HAM);
			
		} else {
			double weighted_sum=w0;
			for (Entry<String, Integer> entry : spamVocabMap.get(file_no).entrySet()) {
				weighted_sum += weighted_word.get(entry.getKey())*entry.getValue();
			}
			return sigmod(weighted_sum, SPAM);
		}
	}
	private static double sigmod(double weighted_sum, String className) {
		if(weighted_sum>100) {
			return 1.0;
		} else if(weighted_sum<-100) {
			return 0.0;
		} else {
			if(className.equals(SPAM)) {
				return (1.0 /(1.0+ Math.exp(weighted_sum)));
			} else {
				return 1-(1.0 /(1.0+ Math.exp(weighted_sum)));
			}
			
		}
	}
	private static void checkWithTest(String testDir) {
		getFiles(testDir,test_allDocs, test_spamDocs, test_hamDocs);	
		readTestData();
		
		int h_count=0, s_count=0;
		/* For ham */
		for (Entry <Integer, LinkedHashMap<String, Integer>> entry : testHamVocabMap.entrySet()) {
			LinkedHashMap<String, Integer> freqMap=entry.getValue();
			int result=testFile(freqMap, HAM);
			if(result == -1) {
				h_count++;
			}
		}
		
		/* For spam */
		for (Entry <Integer, LinkedHashMap<String, Integer>> entry : testSpamVocabMap.entrySet()) {
			LinkedHashMap<String, Integer> freqMap=entry.getValue();
			int result=testFile(freqMap, SPAM);
			if(result == 1) {
				s_count++;
			}
		}
		int accuracy=h_count+s_count;
		System.out.println((double)accuracy*100/((double)test_hamDocs.size()+(double)test_spamDocs.size()));
			
	}
	
	private static int testFile(LinkedHashMap<String, Integer> map,String doc_class) {
		double sum=0;
		for(Entry<String, Integer> entry: map.entrySet()) {
			String word = entry.getKey();
			double freq = (double)entry.getValue();
			if(weighted_word.containsKey(word)) {
				sum += weighted_word.get(word)*freq;
			}
		}
		sum = sum + w0;
		if(sum>=0) {
			return 1;
		} else {
			return -1;
		}
	}

	private static void readTestData() {
		extractSubDirVocab(test_spamDocs, testSpamVocabDir, testSpamVocabMap, "spam");
		extractSubDirVocab(test_hamDocs, testHamVocabDir, testHamVocabMap, "ham");
	}
	private static void getFiles(String directoryName, ArrayList<File> allDocList, ArrayList<File> spamDocList, ArrayList<File> hamDocList) {
	    File spamDirectory = new File(directoryName+"spam/");
	    File hamDirectory = new File(directoryName+"ham/");
	    int n=0;
	    /* Get all the files from a directory */
	    File[] fList = spamDirectory.listFiles();
	    for (File file : fList) {
	        if (file.getName().contains(".txt")) {
	        	/* All docs count */
	        	spamDocList.add(file);
	        	allDocList.add(file);
	        }
	    }
	    fList = hamDirectory.listFiles();
	    for (File file : fList) {
	        if (file.getName().contains(".txt")) {
	        	/* All docs count */
	        	hamDocList.add(file);
	        	allDocList.add(file);
	        }
	    }
		
	}
	private static void readTrainingData() {
		extractAllVocab(allDocs, vocabDir);
		extractSubDirVocab(spamDocs, spamVocabDir, spamVocabMap, "spam");
		extractSubDirVocab(hamDocs, hamVocabDir, hamVocabMap, "ham");
	}
	private static void extractSubDirVocab(ArrayList<File> docs, LinkedHashMap<String, Integer> dir,LinkedHashMap<Integer, LinkedHashMap<String, Integer>> map, String dir_name) {
		for(int i=0;i<docs.size();i++) {
			LinkedHashMap<String, Integer> wordsVocab = new LinkedHashMap<String, Integer>();
			try {
				FileReader file = new FileReader(docs.get(i));
					BufferedReader br = new BufferedReader(file);
					wordsVocab= getWords(br, dir, true);
					
					map.put(doc_count,wordsVocab);
					doc_count++;
		        
			} catch(Exception e) {
				e.printStackTrace();
			}
		}
	}
	private static void extractAllVocab(ArrayList<File> docs, LinkedHashMap<String, Integer> dir) {
		for(int i=0;i<docs.size();i++) {
			LinkedHashMap<String, Integer> wordsVocab = new LinkedHashMap<String, Integer>();
			try {
				FileReader file = new FileReader(docs.get(i));
					BufferedReader br = new BufferedReader(file);
					wordsVocab=getWords(br, dir, true);
			        vocabMap.put(i,wordsVocab);
			} catch(Exception e) {
				e.printStackTrace();
			}
		}
		
	}
	private static LinkedHashMap<String, Integer> getWords(BufferedReader br,LinkedHashMap<String, Integer> vocabdir, boolean calculateDistinct) {
		String line;
		int count =0;
		
		LinkedHashMap<String, Integer> dir = new LinkedHashMap<String, Integer>();
		try{
			while((line = br.readLine()) !=null){
				Pattern pattern = Pattern.compile(REGEX);
				Matcher matcher = pattern.matcher(line);
				
				while(matcher.find()) {
					count++;
					//If word is found increment
					if(dir.containsKey(matcher.group())) {
						for (Entry<String, Integer> entry : dir.entrySet()) {
							if(entry.getKey().equals(matcher.group())) {
								entry.setValue(entry.getValue()+1);
								break;
							}
						}
					} else {
						//Else add the word in the directory
						dir.put(matcher.group(), 1);
					}
					//If word is found increment
					if(vocabdir.containsKey(matcher.group())) {
						for (Entry<String, Integer> entry : vocabdir.entrySet()) {
							if(entry.getKey().equals(matcher.group())) {
								entry.setValue(entry.getValue()+1);
								break;
							}
						}
					} else {
						//Else add the word in the directory
						vocabdir.put(matcher.group(), 1);
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return dir;
	}
}
