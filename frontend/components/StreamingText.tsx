'use client';

import { useEffect, useState } from 'react';

interface StreamingTextProps {
  text: string;
  charDelay?: number;
}

export default function StreamingText({ text, charDelay = 30 }: StreamingTextProps) {
  const [displayedText, setDisplayedText] = useState('');

  useEffect(() => {
    if (!text) return;

    let currentIndex = 0;
    const interval = setInterval(() => {
      if (currentIndex <= text.length) {
        setDisplayedText(text.substring(0, currentIndex));
        currentIndex++;
      } else {
        clearInterval(interval);
      }
    }, charDelay);

    return () => clearInterval(interval);
  }, [text, charDelay]);

  return <>{displayedText}</>;
}
