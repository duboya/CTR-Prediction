import java.util.ArrayList;
import java.util.List;

// Generate GFM support TOC tag
public class TOC {

    private final static String tocTagBefore = "[TOC]";
    private final static String tocTagAfter = "<!-- GFM-TOC -->";
    // catalogue level
    private static int level = 4;


    public static void setCatalogueLevel(int level) {
        TOC.level = level;
    }


    public static List<String> changeTOCToGeneratedCatalogue(List<String> contents) {

        List<String> ret = new ArrayList<String>();

        // If there already existed catalogue of document, then update this catalogue.
        boolean isInOldCatalogue = false;

        // Make sure generate once.
        boolean hasGenerated = false;

        for (String text : contents) {
            if (isInOldCatalogue) {
                if (text.contains(tocTagAfter)) {
                    ret.add(generateCatalogue(contents));
                    hasGenerated = true;
                    isInOldCatalogue = false;
                }
                continue;
            }
            if (hasGenerated) ret.add(text);
            else if (text.contains(tocTagAfter)) {
                isInOldCatalogue = true;
            } else if (text.contains(tocTagBefore)) {
                ret.add(generateCatalogue(contents));
                hasGenerated = true;
            } else ret.add(text);
        }

        return ret;
    }


    private static String generateCatalogue(List<String> contents) {

        List<String> titles = getTitles(contents);

        StringBuilder sb = new StringBuilder();

        sb.append(tocTagAfter).append(CommonUtil.getLineSeparator());

        for (String title : titles) {
            sb.append(formatTitle(title)).append(CommonUtil.getLineSeparator());
        }

        sb.append(tocTagAfter).append(CommonUtil.getLineSeparator());

        return sb.toString();
    }


    private static List<String> getTitles(List<String> contents) {

        List<String> titles = new ArrayList<String>();

        boolean isCode = false;

        for (String line : contents) {
            if (line.contains("```")) {
                isCode = !isCode;
            } else if (line.startsWith("#") && !isCode && getLevelOfATitle(line) <= TOC.level) {
                titles.add(line);
            }
        }

        return titles;
    }


    private static int getLevelOfATitle(String title) {

        int cnt = 0;

        int idx = title.indexOf("#");
        while (idx != -1) {
            cnt++;
            idx = title.indexOf("#", idx + 1);
        }

        return cnt;
    }


    private static String formatTitle(String title) {

        StringBuilder ret = new StringBuilder();

        int level = getLevelOfATitle(title);
        for (int i = 1; i < level; i++) {
            ret.append("    ");
        }

        ret.append("*");
        ret.append(" ");

        int contentIdx = title.lastIndexOf("#");
        contentIdx++;
        while (title.charAt(contentIdx) == ' ') {
            contentIdx++;
        }

        String content = title.substring(contentIdx);

        content = content.trim();

        // replace ' ' to '-'
        String anchor = content.replaceAll(" ", "\\-");

        // remove spacial char
        anchor = anchor.replaceAll("[.：（）()*/、:+]", "");

        // uppercase to lowercase
        anchor = anchor.toLowerCase();

        ret.append("[").append(content).append("]");
        ret.append("(#").append(anchor).append(")");

        return ret.toString();
    }
}
