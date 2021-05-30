import React from "react";
import CTACard from "../components/CTACard";

export default function Introduction() {
  return (
    <>
      <section>
        <h1>Introduction to Decision Trees</h1>
        <p>What are decision trees? What can they be used for?</p>
      </section>
      
      <article>
        <h2>At a high level</h2>
        <p>
          Decision trees provide an easy-to-follow model for making a prediction
          given a set of features. This model is in the form of a tree, and to
          make a prediction, we start at the root node and follow the path given
          the features we want until we reach a leaf node.
        </p>
        <p>
          This is super abstract, so let's tie it back to reality with an
          everyday example!
        </p>
        
        <h3>Choosing your outfit</h3>
        <p>
          Before you hop on Zoom for 5+ hours, you probably have to pick out
          what
          you're going to wear that day. There are many factors that may go into
          this decision including the following
        </p>
        
        <ul>
          <li>temperature <em>(warm or cold)</em></li>
          <li>day of the week <em>(weekday or weekend)</em></li>
          <li>how you feel <em>(energized or lazy)</em></li>
        </ul>
        
        <p>
          As with most decisions in life, a combination of these factors go
          into your final outfit. Depending on how you prioritize these factors,
          you may end up with a decision tree as follows.
        </p>
        
        <img
          src={`${process.env.PUBLIC_URL}/imgs/clothing_tree.png`}
          width="100%"
          alt="decision tree for deciding what to wear"
        />
        
        <p>
          So on a lazy sunday in January, you'd probably be wearing a hoodie and
          sweats.
        </p>
        
        <h2>Creating a tree</h2>
        <p>
          Given a data set and a goal of creating a decision tree, we start by
          finding features that, once a choice is made, reduces the amount of
          entropy within the data set the most.
        </p>
        
        <p>
          Entropy, for our case, is the measure of chaos in our predictions. We
          measure this by looking at the probability of outcomes given a current
          set of data. Our goal when choosing a feature is to minimize the
          overall
          entropy, which is equivalent to maximizing information gained with
          each
          choice. This process is repeated until either no information is gained
          with a choice, or there is only one outcome remaining!
        </p>
        
        <p>Let's see what this looks like with a simple example</p>
        
        <h3>Clothing data set</h3>
        <p>
          For the sake of example, assume that you have been tracking what
          you have been wearing for many days and you find the following results
        </p>
        
        <table width="100%">
          <tr>
            <th>Weather</th>
            <th>Day of week</th>
            <th>Energy</th>
            <th>Clothing</th>
          </tr>
          
          <tr>
            <td>Warm</td>
            <td>Weekday</td>
            <td>Lazy</td>
            <td>T-shirt</td>
          </tr>
          <tr>
            <td>Warm</td>
            <td>Weekend</td>
            <td>Energized</td>
            <td>Hawaiian shirt</td>
          </tr>
          <tr>
            <td>Warm</td>
            <td>Weekend</td>
            <td>Lazy</td>
            <td>Hawaiian shirt</td>
          </tr>
          
          <tr>
            <td>Cold</td>
            <td>Weekday</td>
            <td>Lazy</td>
            <td>Hoodie</td>
          </tr>
          <tr>
            <td>Cold</td>
            <td>Weekday</td>
            <td>Energized</td>
            <td>Jacket</td>
          </tr>
          <tr>
            <td>Cold</td>
            <td>Weekend</td>
            <td>Lazy</td>
            <td>Hoodie</td>
          </tr>
          <tr>
            <td>Cold</td>
            <td>Weekend</td>
            <td>Lazy</td>
            <td>Hoodie</td>
          </tr>
        </table>
        
        <br/>
        <p>
          If we follow the idea that we're trying to reduce entropy (as seen in
          the amount of variance in the "clothing" column), we may go through
          the following process.
        </p>
        
        <ol>
          <li>
            A warm or cold day makes the most difference initially. This is
            how we'll split the dataset.
          </li>
          <li>
            On warm days, we can see that the day of the week makes more of a
            difference than our energy level to what we wear, so we will split
            from here.
            <ul>
              <li>
                Notice that once we make this split, there is no entropy in
                what we will wear, so we make these clothing choices leaf nodes
                on the tree.
              </li>
            </ul>
          </li>
          <li>
            On cold days, we can see that energy level makes a much bigger
            impact
            on our clothing choice than the day of the week, so we will split
            the data set on this feature
            <ul>
              <li>
                Similiarily to warm days, once we make this split we have zero
                entropy, so we are able to make a decision (i.e. produce a leaf
                node)
              </li>
            </ul>
          </li>
        </ol>
        
        <p>
          This process results in a decision tree like the one above! There is
          a little more math involved in the actual production, but this
          recursive process is what we will follow while producing our decision
          tree!
        </p>
        
        <h2>Observations</h2>
        <p>
          We can see from the above example that decision trees are a great way
          to make procedural decisions in an explainable way over discrete data
          sets. Note that there are methods for handling continuous data, namely
          binning.
        </p>
        
        <p>
          We also have to be careful that our tree does not overfit our data set
          by becoming extremely deep (having lots of branches). This may seem
          like a good thing, as it will be accurate over our testing data, but
          this serves little use in making general decisions on new data. We
          can handle this best by limiting the depth of our tree or reducing
          the amount of features we use.
        </p>
      </article>
      <CTACard
        title="Practice time"
        description="If you're ready, see how to get hands on experience with some examples"
        buttonText="Getting Started" linkTo="/#/getting-started"
        secondary={true}
      />
    
    </>
  );
}
