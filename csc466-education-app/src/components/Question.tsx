import { FormControlLabel, Paper, Switch, Typography } from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import React, { useState } from "react";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    wrapper: {
      width: "80%",
      margin: "10px auto",
      padding: "20px"
    },
    answer: {
      margin: "10px auto",
      padding: "20px",
      backgroundColor: "whitesmoke"
    }
  })
);

export interface QuestionProps {
  question: string,
  answer: string
}

export default function Question(props: QuestionProps) {
  const classes = useStyles();
  const [showAnswer, setShowAnswer] = useState(false);
  
  return (
    <Paper className={classes.wrapper} variant={"outlined"}>
      <div style={{display: "flex", justifyContent: "space-between"}}>
        <Typography variant={"h6"}>
          {props.question}
        </Typography>
        <FormControlLabel
          control={
            <Switch
              checked={showAnswer}
              onChange={() => setShowAnswer(!showAnswer)}
              name="showAnswer"/>}
          label="Show Answer"
        />
      </div>
      {showAnswer &&
      <Paper className={classes.answer} variant={"outlined"}>
        <Typography>
          {props.answer}
        </Typography>
      </Paper>}
    </Paper>
  );
}
