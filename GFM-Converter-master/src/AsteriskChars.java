public class AsteriskChars {

    public static String addSpace(String text) {

        int startIdx = text.indexOf("**");
        if (startIdx == -1) return text;

        int endIdx = text.indexOf("**", startIdx + 3);
        if (endIdx == -1) return text;

        StringBuilder ret = new StringBuilder();

        if (startIdx != 0) {
            ret.append(text, 0, startIdx);
            ret.append(" ");
        }

        ret.append(text, startIdx, endIdx + 2);

        if (endIdx != text.length() - 1) {
            ret.append(" ");
            ret.append(text.substring(endIdx + 2));
        }

        return ret.toString();
    }
}
