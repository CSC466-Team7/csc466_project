import React from "react";
import { Link } from "react-router-dom";

import { Button } from "@material-ui/core";
import CTACard from "../../components/CTACard";

export default function Introduction() {
  return (
    <>
      <section>
        <h1>Example Holder</h1>
        <p>tag line</p>
      </section>

      <article>
        <h2>more info here</h2>
        <p>maybe some details about the goal of this example</p>
      </article>

      {/*<CTACard>*/}
      {/*  <span>*/}
      {/*    <h2>Download example</h2>*/}
      {/*    <p>*/}
      {/*      To follow along download the example. Need&nbsp;*/}
      {/*      <Link to="../getting-started">help</Link>?*/}
      {/*    </p>*/}
      {/*  </span>*/}
      {/*  <Button*/}
      {/*    component="a"*/}
      {/*    color="primary"*/}
      {/*    variant="contained"*/}
      {/*    href={`${process.env.PUBLIC_URL}/test.txt`}*/}
      {/*    download*/}
      {/*  >*/}
      {/*    Download*/}
      {/*  </Button>*/}
      {/*</CTACard>*/}

    </>
  );
}
