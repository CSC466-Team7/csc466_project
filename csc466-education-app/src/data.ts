export const tutorials = [
  /*Decision Tree Algorithm*/
  {
    img: {
      url: "https://miro.medium.com/max/781/1*fGX0_gacojVa6-njlCrWZw.png",
      title: "Decision Tree Visual"
    },
    content: {
      title: "Decision Tree - ID3",
      description:
        "Learn how to implement a decision tree from scratch with Python",
      dataset: "heart.csv",
      questions: [
        {
          question: "Using ID3 tree creation, assuming 3 features are left, how many more iterations, including this one and the base case, will it take to hit and return from a base case for creating the tree, assuming each feature, even after being split up, has at least 2 unique values per class (e.g. Class A of Feature 3 would still have 2 unique values) and all information gains are 0 no matter what?",
          answer: "4. We'll keep going until we are out of features, so 1 for each feature and then 1 last iteration for the base case."
        },
        {
          question: "If we returned the highest frequency class in our target when the max information gain from any column was 0, would our accuracy scores for VALIDATION DATA be able to decrease, increase, or stay the same? Select all that apply.",
          answer: "The scores do any of the above.\n" +
          "- Stay the same: Each branch we take after we have no more info gain may yield the same result as picking the highest frequency class.\n" +
          "- Increase: An IG of 0 means there was no change in entropy. Branching again could cause entropy to change with a remaining feature (entropy can only decrease here)\n" +
          "- Decrease: The opposite of the above"
        }
      ]
    },
    notebook: "heart_decision_tree_classifier"
  },
  /*Decision Tree Algorithm (C4.5)*/
  {
    img: {
      url: "https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2015/01/Decision-Tree-Example-6-Decision-tree-Edureka.png",
      title: "Decision Tree Visual"
    },
    content: {
      title: "Decision Tree - C4.5",
      description:
        "Implement decision trees from scratch using the C4.5 algorithm, which handles continuous features. Also included: min_split_count",
      dataset: "heart.csv",
      questions: [
        {
          question: "How does the C4.5 algorithm handle features with continuous values when choosing a variable to split on?",
          answer: "When we encounter a continuous variable while choosing a feature to split on, we iterate through all possible splits between the values in this feature. At each point, we calculate the information gained if a split was made at this point. After calculating all possible splits and the associated information gained, we choose the split that yeilds us the most information gained. At this point, we treat this column as a categorical feature based on where the split was and then continue with the normal decision tree creation.\n Note: We do not modify the actual data after finding a split, this is only used for creating the tree."
        },
        {
          question: "How can you determine a good value for min_split_count?",
          answer: "This is slightly dependent on the size of your dataset. With larger datasets, you can have higher min_split_counts that will still produce accurate trees, while smaller datasets may require smaller values (as each split will significantly reduce the amount of items remaining). You can use a testing-based approach to determine an optimal value by building trees with differing values for the min_split_count and comparing their performance using a testing dataset."
        }
      ]
    },
    notebook: "heart_decision_tree_c45"
  },
  /*Scikit Learn Classifier*/
  {
    img: {
      url: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png",
      title: "Scitkit Learn Logo"
    },
    content: {
      title: "Scikit Learn Classifier",
      description:
        "Using the decision tree classifier we implemented in the previous" +
        " tutorial, we will now transform that into a classifier that works" +
        " with sklearn",
      dataset: "heart.csv",
      questions: [
        {
          question: "Why do you think `check_estimator` has no output if it succeeded? What might it do if it failed?",
          answer: "`check_estimator` is basically doing unit testing on your estimator. When it fails, an error is raised. Thus no errors raised means all the checks succeeded."
        }
      ]
    },
    notebook: "heart_classifier_with_sklearn"
  },
  /*Bagging and Random Forests*/
  {
    img: {
      url: "https://miro.medium.com/max/1052/1*VHDtVaDPNepRglIAv72BFg.jpeg",
      title: "Random Forests"
    },
    content: {
      title: "Bagging and Random Forests",
      description:
        "Using what we know about decision trees, we will now learn how to" +
        " make more powerful models by leveraging bagging and random forests",
      questions: [
        {
          question: "What is the difference between Bagging and Random Forest?",
          answer: "The primary difference between Random Forest and Bagging" +
            " is that for Random Forest, only a random subset of features" +
            " are used to create a tree, whereas in Bagging, the full set of" +
            " features are used."
        },
        {
          question: "In general, would an ensemble model perform better with" +
            " 2 tress or 25 trees?",
          answer: "It will perform better with 25 trees."
        },
        {
          question: "What are some advantages of Random Forest?",
          answer: "It helps to improve the accuracy of the model, especially" +
            " when there is missing or insufficient data. Additionally," +
            " it reduces overfitting in decision trees since it accounts for" +
            " variance in data by training on varied samples of the" +
            " training data."
        }
      ]
    },
    notebook: "bagging_and_random_forest"
  }];

export const skills = [
  {
    img: {
      url: "https://www.megahowto.com/wp-content/uploads/2009/09/Rubix-Cube.jpg",
      title: "a rubix cube"
    },
    content: {
      title: "Numpy",
      description:
        "A library for easily handling multi-dimensional arrays and matrices"
    },
    linkTo: "https://numpy.org/doc/stable/user/absolute_beginners.html"
  },
  {
    img: {
      url: "https://cameoglassuk.co.uk/wp-content/uploads/2016/07/EATING-PANDAS-1.jpg",
      title: "pandas eating bamboo"
    },
    content: {
      title: "Pandas",
      description:
        "A library for powerful data analysis and manipulation."
    },
    linkTo: "https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html"
  },
  {
    img: {
      url: "https://miro.medium.com/max/1000/1*0DDt5Xp9z6ecj5eL6FNAfQ.png",
      title: "data clustering example"
    },
    content: {
      title: "Scikit Learn",
      description:
        "A library that makes ML processes (like dimensionality reduction)" +
        " easy."
    },
    linkTo: "https://scikit-learn.org/stable/tutorial/index.html"
  }
];
