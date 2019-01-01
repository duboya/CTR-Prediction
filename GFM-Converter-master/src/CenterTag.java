public class CenterTag {

    public static String convert(String text) {

        boolean containsCenterTag = text.contains("<center>") && text.contains("</center>");
        if (!containsCenterTag) return text;

        int startIdx = text.indexOf("<center>");
        int endIdx = text.indexOf("</center>");
        text = text.substring(startIdx + 8, endIdx);

        boolean hasPic = text.contains("![]");
        if (!hasPic) return text;

        startIdx = text.indexOf("![]");
        endIdx = text.indexOf(")", startIdx);
        String picPath = text.substring(startIdx + 4, endIdx);
        text = "<img src=\"" + picPath + "\"/>";
        text = "<br><div align=\"center\"> " + text + " </div><br>";

        return text;
    }
}
