import { FormControlLabel, Paper, Switch, Typography } from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import React, { useState } from "react";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    wrapper: {
      width: "80%",
      margin: "10px auto",
      padding: "20px",
      position: "relative",
    },
    header: {
      maxWidth: "80%",
    },
    question: {
      maxWidth: "80%",
    },
    answer: {
      margin: "10px auto",
      padding: "20px",
      backgroundColor: "whitesmoke"
    },
    toggleControl: {
      position: "absolute",
      right: "0",
      top: "20px",
      maxWidth: "20%",
    },
  })
);

export interface TextOnlyQuestionProps {
  question: string,
  answer: string
}
export function TextOnlyQuestion(props: TextOnlyQuestionProps) {
  return (
    <Question
      question={
        <Typography variant={"h6"}>
          {props.question}
        </Typography>
      }

      answer={
        <Typography>
          {props.answer}
        </Typography>
      }
    />
  );
}

export interface QuestionProps {
  question: string | JSX.Element,
  answer: string | JSX.Element,
  header?: JSX.Element,
  styleOverrides?: Partial<React.CSSProperties>,
}

export default function Question(props: QuestionProps) {
  const classes = useStyles();
  const [showAnswer, setShowAnswer] = useState(false);
  
  return (
    <Paper style={props.styleOverrides} className={classes.wrapper} variant={"outlined"}>
      <div className={classes.header}>
        {props.header}
      </div>
      <FormControlLabel
        className={classes.toggleControl}
        control={
          <Switch
            checked={showAnswer}
            onChange={() => setShowAnswer(!showAnswer)}
            name="showAnswer"/>}
        label="Show Answer"
      />
      <div className={classes.question}>
        {props.question}
      </div>
      {showAnswer &&
      <Paper className={classes.answer} variant={"outlined"}>
        {props.answer}
      </Paper>}
    </Paper>
  );
}
