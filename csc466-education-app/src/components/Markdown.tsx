import { IconButton, Paper } from "@material-ui/core";
import { FileCopy } from "@material-ui/icons";
import React, { useEffect, useState } from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import dark from "react-syntax-highlighter/dist/esm/styles/prism/material-dark";
import { ASSETS_URL } from "../constants";

export default function Markdown(props: { fileName: string; }) {
  const [markdownAsString, setMarkdownAsString] = useState("");
  const fetchMarkdown = async () => {
    const res = await fetch(`${ASSETS_URL}/${props.fileName}`);
    const resText = await res.text();
    setMarkdownAsString(resText);
  };
  useEffect(() => {
    fetchMarkdown();
  }, []);
  const components: any = {
    // @ts-ignore
    code({node, inline, className, children, ...props}) {
      const match = /language-(\w+)/.exec(className || "");
      return !inline && match ? (
        <span>
          <CopyToClipboard text={String(children)}>
            <IconButton color="primary"
              component="span"
              style={{float: "right", marginLeft: 10}}>
              <FileCopy/>
            </IconButton>
          </CopyToClipboard>
          <SyntaxHighlighter style={dark} language={match[1]} PreTag="div"
            children={
              String(children)
                .replace(/\n$/, "")
            }
            {...props} />
        </span>
         
      ) : (
        <code className={className} children={children} {...props} />
      );
    }
  };
  return (
    <Paper elevation={3} style={{padding: 20, margin: 20}}>
      <article className="markdown-body">
        <ReactMarkdown
          components={components}>{markdownAsString}</ReactMarkdown>
      </article>
    </Paper>
  );
}