import React from "react";
import { Link } from "react-router-dom";
import Article from "../components/Article";
import CTACard from "../components/CTACard";
import Header from "../components/Header";

export default function Introduction() {
  return (
    <>
      <Header
        title={"Getting Started"}
        description={"Here's what you'll need to follow along with the tutorials"}
      />
  
      <Article>
        <h2>Learning about Decision Trees</h2>
        <p>
          The <Link to="tutorials">tutorials</Link> on this site are set up in
          such a way that, by following along, you will learn many of the
          concepts behind decision trees. This is reflective of Cal Poly's
          "Learn by Doing" methodology, which (through personal observation)
          works quite beautifully.
        </p>
        
        <p>
          However, this style of learning is really only helpful if you have
          a basic understanding of the concepts you are trying to learn. If
          you're completely unfamiliar with decision trees, you should&nbsp;
          <Link to="introduction">start here</Link> to get a high level
          overview of what we're trying to accomplish with these. Otherwise,
          keep on reading to see how to follow along with the tutorials!
        </p>
        
        <h2>Running a tutorial</h2>
        <p>
          Before we're able to work on an actual example, you must first&nbsp;
          <a
            href="https://jupyter.org/install#getting-started-with-the-classic-jupyter-notebook"
            target="_blank"
            rel="noreferrer"
          >
            install Jupyter Notebook
          </a>. After you have successfully done that, download an example's
          "starting file", and open it using Jupyter notebook.
        </p>
        
        <p>
          The starting files provide a clear learning path with helpful comments
          to guide you down the right path while building working decision
          trees!
        </p>
      </Article>
      <CTACard
        title={"Working with Tutorials"}
        description="Now that you know how to get them going, it's time to get your hands dirty with code!"
        buttonText="Tutorials" linkTo="/tutorials"
        secondary={true}
      />
    
    </>
  );
}
