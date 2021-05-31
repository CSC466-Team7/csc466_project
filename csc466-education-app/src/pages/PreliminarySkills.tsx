import React from "react";
import CustomCard from "../components/CustomCard";

import Gallery from "../components/Gallery";

export default function PreliminarySkills() {
  const skills = [
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
  
  return (
    <>
      <section>
        <h2>Preliminary Skills</h2>
        <p>
          Need a quick brush up on some of the KDD tools used to build decesion
          trees? Look no further!<br/>
          Here are links to tutorials of some commonly used tools in data
          science.
        </p>
      </section>
      
      <Gallery>
        {skills.map((skill =>
          <CustomCard {...skill} key={skill.content.title}/>
        ))}
      </Gallery>
      
      <section>
        <p>
          While these won't cover everything you'll need to know, hopefully
          they can serve as a nice refresher.
        </p>
      </section>
    </>
  );
}
