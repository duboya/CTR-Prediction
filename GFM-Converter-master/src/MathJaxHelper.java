public class MathJaxHelper {

    public static String changeMathJaxToCodeCogs(String text) {

        StringBuilder sb = new StringBuilder();

        while (true) {

            int startIdx, endIdx;
            if (isLineMath(text)) {
                startIdx = text.indexOf("$$");
                endIdx = text.indexOf("$$", startIdx + 1);
            } else if (isInLineMath(text)) {
                startIdx = text.indexOf("$");
                endIdx = text.indexOf("$", startIdx + 1);
            } else {
                sb.append(text);
                break;
            }

            String leftPartContent = text.substring(0, startIdx);

            while (startIdx < text.length() && text.charAt(startIdx) == '$') startIdx++;
            String mathJaxContent = text.substring(startIdx, endIdx);
            mathJaxContent = mathJaxContent.replaceAll(" ", "");
            mathJaxContent = "<img src=\"https://latex.codecogs.com/gif.latex?" + mathJaxContent + "\"/>";

            if (isLineMath(text)) {
                mathJaxContent = "<div align=\"center\">" + mathJaxContent + "</div> <br>";
            }

            while (endIdx < text.length() && text.charAt(endIdx) == '$') endIdx++;
            sb.append(leftPartContent).append(mathJaxContent);
            text = text.substring(endIdx);
        }

        return sb.toString();
    }


    private static boolean isLineMath(String text) {
        return hasPairs(text, "$$");
    }


    private static boolean isInLineMath(String text) {
        return hasPairs(text, "$");
    }


    private static boolean hasPairs(String text, String str) {

        int idx = text.indexOf(str);
        if (idx == -1 || (idx != 0 && text.charAt(idx - 1) == '\\')) {
            return false;
        }
        idx = text.indexOf(str, idx + 3);
        return idx != -1;
    }
}
