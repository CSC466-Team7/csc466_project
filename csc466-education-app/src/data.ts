export const tutorials = [
  /*Decision Tree Algorithm*/
  {
    img: {
      url: "https://miro.medium.com/max/781/1*fGX0_gacojVa6-njlCrWZw.png",
      title: "Decision Tree Visual"
    },
    content: {
      title: "Decision Tree Algorithm",
      description:
        "Learn how to implement a decision tree from scratch with Python",
      dataset: "heart.csv",
      questions: [
        {
          question: "What is a cat?",
          answer: "An animal"
        },
        {
          question: "Where does wood come from?",
          answer: "Trees"
        }
      ]
    },
    notebook: "heart_decision_tree_classifier"
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
          question: "What is a cat?",
          answer: "An animal"
        },
        {
          question: "Where does wood come from?",
          answer: "Trees"
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